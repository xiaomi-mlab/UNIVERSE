import json
import math
import os
import time
from datetime import timedelta
from typing import Optional

import torch
from tqdm import tqdm

from ..models.utils import load_state_dict
from ..utils import ModelConfig


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._ema_enabled = False
        self._ema_decay = 0.999
        self._ema_update_after_step = 0
        self._ema_update_every = 1
        self._ema_named_params = []
        self._ema_shadow = {}

    def to(self, *args, **kwargs):
        for _, model in self.named_children():
            model.to(*args, **kwargs)
        return self

    def trainable_modules(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def trainable_param_names(self):
        return {name for name, param in self.named_parameters() if param.requires_grad}

    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        try:
            from peft import LoraConfig, inject_adapter_in_model
        except ImportError as exc:
            raise RuntimeError("peft is required when --lora_base_model is enabled.") from exc

        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model

    @staticmethod
    def mapping_lora_state_dict(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace(
                    "lora_B.weight", "lora_B.default.weight"
                )
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_names}
        if remove_prefix is None:
            return state_dict
        return {
            (name[len(remove_prefix) :] if name.startswith(remove_prefix) else name): param
            for name, param in state_dict.items()
        }

    def init_ema(
        self,
        enabled=True,
        decay=0.999,
        device="cpu",
        update_after_step=0,
        update_every=1,
    ):
        self._ema_enabled = bool(enabled)
        self._ema_shadow = {}
        self._ema_named_params = []
        if not self._ema_enabled:
            return

        decay = float(decay)
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
        ema_device = torch.device(device)
        self._ema_decay = decay
        self._ema_update_after_step = max(int(update_after_step), 0)
        self._ema_update_every = max(int(update_every), 1)
        self._ema_named_params = [(name, param) for name, param in self.named_parameters() if param.requires_grad]
        with torch.no_grad():
            for name, param in self._ema_named_params:
                self._ema_shadow[name] = param.detach().to(device=ema_device, dtype=torch.float32).clone()

    def has_ema(self):
        return bool(self._ema_enabled and len(self._ema_shadow) > 0)

    def _should_update_ema(self, step_id):
        step_id = int(step_id)
        if step_id <= self._ema_update_after_step:
            return False
        return (step_id - self._ema_update_after_step - 1) % self._ema_update_every == 0

    def update_ema(self, step_id):
        if not self.has_ema() or not self._should_update_ema(step_id):
            return False
        decay = float(self._ema_decay)
        with torch.no_grad():
            for name, param in self._ema_named_params:
                shadow = self._ema_shadow.get(name)
                if shadow is None:
                    continue
                source = param.detach()
                if source.device != shadow.device or source.dtype != torch.float32:
                    source = source.to(device=shadow.device, dtype=torch.float32)
                shadow.mul_(decay).add_(source, alpha=1.0 - decay)
        return True

    def export_ema_trainable_state_dict(self, remove_prefix=None):
        if not self.has_ema():
            return {}
        state_dict = {}
        with torch.no_grad():
            for name, param in self._ema_named_params:
                shadow = self._ema_shadow.get(name)
                if shadow is None:
                    continue
                value = shadow.to(dtype=param.dtype, device="cpu").clone()
                export_name = name[len(remove_prefix) :] if remove_prefix and name.startswith(remove_prefix) else name
                state_dict[export_name] = value
        return state_dict

    def parse_model_configs(self, model_paths, model_id_with_origin_paths, enable_fp8_training=False):
        offload_dtype = torch.float8_e4m3fn if enable_fp8_training else None
        model_configs = []
        if model_paths is not None:
            model_configs += [ModelConfig(path=path, offload_dtype=offload_dtype) for path in json.loads(model_paths)]
        if model_id_with_origin_paths is not None:
            for item in model_id_with_origin_paths.split(","):
                model_id, origin = item.split(":", 1)
                model_configs.append(ModelConfig(model_id=model_id, origin_file_pattern=origin, offload_dtype=offload_dtype))
        return model_configs

    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models,
        lora_base_model,
        lora_target_modules,
        lora_rank,
        lora_checkpoint=None,
        enable_fp8_training=False,
    ):
        pipe.scheduler.set_timesteps(1000, training=True)
        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))

        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=[name.strip() for name in lora_target_modules.split(",") if name.strip()],
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            if lora_checkpoint is not None:
                state_dict = self.mapping_lora_state_dict(load_state_dict(lora_checkpoint))
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, unexpected LoRA keys: {load_result[1]}")
            setattr(pipe, lora_base_model, model)


class ModelLogger:
    def __init__(
        self,
        output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=lambda x: x,
        save_raw_ckpt=True,
        save_ema_ckpt=False,
        ema_file_suffix="-ema",
    ):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.save_raw_ckpt = bool(save_raw_ckpt)
        self.save_ema_ckpt = bool(save_ema_ckpt)
        self.ema_file_suffix = str(ema_file_suffix)
        self.num_steps = 0

    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % int(save_steps) == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def on_epoch_end(self, accelerator, model, epoch_id):
        self.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors")

    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % int(save_steps) != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            if self.save_raw_ckpt:
                state_dict = accelerator.get_state_dict(model)
                state_dict = unwrapped_model.export_trainable_state_dict(
                    state_dict,
                    remove_prefix=self.remove_prefix_in_ckpt,
                )
                state_dict = self.state_dict_converter(state_dict)
                accelerator.save(state_dict, os.path.join(self.output_path, file_name), safe_serialization=True)
            if self.save_ema_ckpt:
                if hasattr(unwrapped_model, "has_ema") and unwrapped_model.has_ema():
                    stem, ext = os.path.splitext(file_name)
                    ema_name = f"{stem}{self.ema_file_suffix}{ext}" if ext else f"{file_name}{self.ema_file_suffix}"
                    ema_state = unwrapped_model.export_ema_trainable_state_dict(
                        remove_prefix=self.remove_prefix_in_ckpt,
                    )
                    ema_state = self.state_dict_converter(ema_state)
                    accelerator.save(ema_state, os.path.join(self.output_path, ema_name), safe_serialization=True)
                else:
                    print("[train][ema][warn] save_ema_ckpt enabled but EMA is not initialized; skip.")
        accelerator.wait_for_everyone()


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().item())
    return float(value)


def _lr_factor(step_index, total_steps, scheduler_type, warmup_steps, warmup_start_factor):
    if warmup_steps > 0 and step_index < warmup_steps:
        if warmup_steps == 1:
            progress = 1.0
        else:
            progress = step_index / float(warmup_steps - 1)
        return warmup_start_factor + (1.0 - warmup_start_factor) * progress
    progress = max(0.0, min(1.0, (step_index - warmup_steps) / float(max(total_steps - warmup_steps - 1, 1))))
    if scheduler_type == "cosine":
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    if scheduler_type == "linear":
        return 1.0 - 0.9 * progress
    return 1.0


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    args=None,
):
    try:
        from accelerate import Accelerator
        from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
    except ImportError as exc:
        raise RuntimeError("accelerate is required for UNIVERSE training. Install requirements.txt first.") from exc

    if args is not None:
        learning_rate = float(args.learning_rate)
        weight_decay = float(args.weight_decay)
        num_workers = int(args.dataset_num_workers)
        save_steps = args.save_steps
        num_epochs = int(args.num_epochs)
        gradient_accumulation_steps = int(args.gradient_accumulation_steps)
        find_unused_parameters = bool(args.find_unused_parameters)

    gradient_accumulation_steps = max(int(gradient_accumulation_steps), 1)
    ddp_timeout_seconds = int(getattr(args, "ddp_timeout_seconds", os.environ.get("NCCL_TIMEOUT", "1800"))) if args is not None else int(os.environ.get("NCCL_TIMEOUT", "1800"))
    log_every_steps = int(getattr(args, "log_every_steps", 100)) if args is not None else 100
    warmup_steps = max(0, int(getattr(args, "warmup_steps", 0))) if args is not None else 0
    warmup_start_factor = float(getattr(args, "warmup_start_factor", 0.01)) if args is not None else 0.01
    warmup_start_factor = min(max(warmup_start_factor, 0.0), 1.0)
    scheduler_type = getattr(args, "lr_scheduler_type", "cosine") if args is not None else "cosine"
    gradient_clip_norm = getattr(args, "gradient_clip_norm", None) if args is not None else None
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
    use_ema = bool(getattr(args, "use_ema", False)) if args is not None else False

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    dataloader_kwargs = dict(
        shuffle=True,
        collate_fn=lambda x: x[0],
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    updates_per_epoch = max(math.ceil(len(dataloader) / gradient_accumulation_steps), 1)
    total_steps = max(num_epochs * updates_per_epoch, 1)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters),
            InitProcessGroupKwargs(timeout=timedelta(seconds=max(ddp_timeout_seconds, 1))),
        ],
    )
    if accelerator.is_main_process:
        print(
            "[train] setup:",
            f"samples={len(dataset)}",
            f"epochs={num_epochs}",
            f"steps={total_steps}",
            f"workers={num_workers}",
            f"lr={learning_rate}",
            f"lr_scheduler={scheduler_type}",
            f"warmup_steps={warmup_steps}",
            f"grad_accum={gradient_accumulation_steps}",
        )

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    unwrapped_model = accelerator.unwrap_model(model)
    if hasattr(unwrapped_model, "init_ema"):
        if use_ema:
            ema_device = "cpu" if bool(getattr(args, "ema_on_cpu", False)) else str(accelerator.device)
            unwrapped_model.init_ema(
                enabled=True,
                decay=float(getattr(args, "ema_decay", 0.999)),
                device=ema_device,
                update_after_step=int(getattr(args, "ema_update_after_step", 0)),
                update_every=int(getattr(args, "ema_update_every", 1)),
            )
            if accelerator.is_main_process:
                print(f"[train][ema] enabled device={ema_device}")
        else:
            unwrapped_model.init_ema(enabled=False)

    step_id = 0
    optimizer.zero_grad()
    last_log = time.perf_counter()
    for epoch_id in range(num_epochs):
        progress = tqdm(dataloader, disable=not accelerator.is_main_process, desc=f"epoch {epoch_id}")
        for data in progress:
            with accelerator.accumulate(model):
                current_lr = learning_rate * _lr_factor(
                    step_id,
                    total_steps,
                    scheduler_type,
                    warmup_steps,
                    warmup_start_factor,
                )
                for group in optimizer.param_groups:
                    group["lr"] = current_lr

                output = model(data)
                if isinstance(output, dict):
                    loss = output["loss"]
                    video_loss = _to_float(output.get("video_loss"))
                    traj_loss = _to_float(output.get("trajectory_loss"))
                else:
                    loss = output
                    video_loss = None
                    traj_loss = None

                accelerator.backward(loss)
                if gradient_clip_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    step_id += 1
                    if use_ema and hasattr(unwrapped_model, "update_ema"):
                        unwrapped_model.update_ema(step_id)
                    model_logger.on_step_end(accelerator, model, save_steps)

                    if accelerator.is_main_process and log_every_steps > 0 and step_id % log_every_steps == 0:
                        elapsed = time.perf_counter() - last_log
                        last_log = time.perf_counter()
                        print(
                            f"[train][step {step_id}/{total_steps}] "
                            f"loss={_to_float(loss):.6f} "
                            f"video_loss={video_loss if video_loss is not None else 'n/a'} "
                            f"traj_loss={traj_loss if traj_loss is not None else 'n/a'} "
                            f"lr={current_lr:.6e} "
                            f"elapsed_s={elapsed:.2f}"
                        )

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)
