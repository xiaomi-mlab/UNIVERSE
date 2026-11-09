from typing import List, Optional, Sequence, Union

import torch

from .longcat_video_dit import LongCatVideoTransformer3DModel
from .utils import hash_state_dict_keys, init_weights_on_device, load_state_dict
from .wan_video_dit import WanModel
from .wan_video_image_encoder import WanImageEncoder
from .wan_video_text_encoder import WanTextEncoder
from .wan_video_vae import WanVideoVAE, WanVideoVAE38


_WAN_MODEL_CONFIGS = {
    # Wan DiT variants used by Wan 2.1/2.2 video checkpoints.
    "9269f8db9040a9d860eaca435be61814": ("wan_video_dit", WanModel, "civitai"),
    "aafcfd9672c3a2456dc46e1cb6e52c70": ("wan_video_dit", WanModel, "civitai"),
    "6bfcfb3b342cb286ce886889d519a77e": ("wan_video_dit", WanModel, "civitai"),
    "6d6ccde6845b95ad9114ab993d917893": ("wan_video_dit", WanModel, "civitai"),
    "349723183fc063b2bfc10bb2835cf677": ("wan_video_dit", WanModel, "civitai"),
    "efa44cddf936c70abd0ea28b6cbe946c": ("wan_video_dit", WanModel, "civitai"),
    "3ef3b1f8e1dab83d5b71fd7b617f859f": ("wan_video_dit", WanModel, "civitai"),
    "70ddad9d3a133785da5ea371aae09504": ("wan_video_dit", WanModel, "civitai"),
    "26bde73488a92e64cc20b0a7485b9e5b": ("wan_video_dit", WanModel, "civitai"),
    "ac6a5aa74f4a0aab6f64eb9a72f19901": ("wan_video_dit", WanModel, "civitai"),
    "b61c605c2adbd23124d152ed28e049ae": ("wan_video_dit", WanModel, "civitai"),
    "1f5ab7703c6fc803fdded85ff040c316": ("wan_video_dit", WanModel, "civitai"),
    "5b013604280dd715f8457c6ed6d6a626": ("wan_video_dit", WanModel, "civitai"),
    "2267d489f0ceb9f21836532952852ee5": ("wan_video_dit", WanModel, "civitai"),
    "5ec04e02b42d2580483ad69f4e76346a": ("wan_video_dit", WanModel, "civitai"),
    "47dbeab5e560db3180adf51dc0232fb1": ("wan_video_dit", WanModel, "civitai"),
    "cb104773c6c2cb6df4f9529ad5c60d0b": ("wan_video_dit", WanModel, "diffusers"),
    # Wan2.2 TI2V-5B LongCat DiT.
    "8b27900f680d7251ce44e2dc8ae1ffef": ("wan_video_dit", LongCatVideoTransformer3DModel, "civitai"),
    # Shared Wan runtime components.
    "9c8818c2cbea55eca56c7b447df170da": ("wan_video_text_encoder", WanTextEncoder, "civitai"),
    "5941c53e207d62f20f9025686193c40b": ("wan_video_image_encoder", WanImageEncoder, "civitai"),
    "1378ea763357eea97acdef78e65d6d96": ("wan_video_vae", WanVideoVAE, "civitai"),
    "ccc42284ea13e1ad04693284c7a09be6": ("wan_video_vae", WanVideoVAE, "civitai"),
    "e1de6c02cdac79f8b739f4d3698cd216": ("wan_video_vae", WanVideoVAE38, "civitai"),
}


def _load_state_dict_from_paths(file_path: Union[str, Sequence[str]]) -> dict:
    if isinstance(file_path, (list, tuple)):
        state_dict = {}
        for path in file_path:
            state_dict.update(load_state_dict(str(path)))
        return state_dict
    return load_state_dict(str(file_path))


def _guess_model_config(file_path: Union[str, Sequence[str]], state_dict: dict):
    keys_hash = hash_state_dict_keys(state_dict, with_shape=True)
    if keys_hash in _WAN_MODEL_CONFIGS:
        return _WAN_MODEL_CONFIGS[keys_hash]

    if isinstance(file_path, (list, tuple)):
        name = " ".join(str(p).lower() for p in file_path)
    else:
        name = str(file_path).lower()
    if "models_t5" in name or "umt5" in name:
        return "wan_video_text_encoder", WanTextEncoder, "civitai"
    if "vae" in name:
        return "wan_video_vae", WanVideoVAE, "civitai"
    if "clip" in name:
        return "wan_video_image_encoder", WanImageEncoder, "civitai"
    if "diffusion" in name or "dit" in name:
        return "wan_video_dit", WanModel, "civitai"
    return None


def _convert_and_build_model(state_dict: dict, model_class: type, model_resource: str, torch_dtype, device):
    converter = model_class.state_dict_converter()
    if model_resource == "diffusers":
        converted = converter.from_diffusers(state_dict)
    else:
        converted = converter.from_civitai(state_dict)

    if isinstance(converted, tuple):
        model_state_dict, extra_kwargs = converted
    else:
        model_state_dict, extra_kwargs = converted, {}

    dtype = torch.float32 if extra_kwargs.get("upcast_to_float32", False) else torch_dtype
    with init_weights_on_device():
        model = model_class(**extra_kwargs)
    model.eval()
    model.load_state_dict(model_state_dict, assign=True)
    return model.to(dtype=dtype, device=device)


class ModelManager:
    def __init__(
        self,
        torch_dtype=torch.float16,
        device="cuda",
        file_path_list: Optional[List[str]] = None,
        **_: object,
    ):
        self.torch_dtype = torch_dtype
        self.device = device
        self.model = []
        self.model_path = []
        self.model_name = []
        self.load_models(file_path_list or [])

    def load_model(self, file_path, model_names=None, device=None, torch_dtype=None):
        print(f"Loading models from: {file_path}")
        device = self.device if device is None else device
        torch_dtype = self.torch_dtype if torch_dtype is None else torch_dtype
        state_dict = _load_state_dict_from_paths(file_path)
        model_config = _guess_model_config(file_path, state_dict)
        if model_config is None:
            print("    Unsupported checkpoint type in UNIVERSE inference runtime. No models are loaded.")
            return

        model_name, model_class, model_resource = model_config
        if model_names is not None and model_name not in model_names:
            print(f"    Detected {model_name}, skipped by model_names filter.")
            return

        print(f"    model_name: {model_name} model_class: {model_class.__name__}")
        model = _convert_and_build_model(state_dict, model_class, model_resource, torch_dtype, device)
        self.model.append(model)
        self.model_path.append(file_path)
        self.model_name.append(model_name)
        print(f"    The following models are loaded: {[model_name]}.")

    def load_models(self, file_path_list, model_names=None, device=None, torch_dtype=None):
        for file_path in file_path_list:
            self.load_model(file_path, model_names=model_names, device=device, torch_dtype=torch_dtype)

    def fetch_model(self, model_name, file_path=None, require_model_path=False, index=None):
        fetched_models = []
        fetched_model_paths = []
        for model, model_path, model_name_ in zip(self.model, self.model_path, self.model_name):
            if file_path is not None and file_path != model_path:
                continue
            if model_name == model_name_:
                fetched_models.append(model)
                fetched_model_paths.append(model_path)
        if len(fetched_models) == 0:
            print(f"No {model_name} models available.")
            return None
        if len(fetched_models) == 1:
            print(f"Using {model_name} from {fetched_model_paths[0]}.")
            model = fetched_models[0]
            path = fetched_model_paths[0]
        elif isinstance(index, int):
            model = fetched_models[:index]
            path = fetched_model_paths[:index]
            print(f"Using {model_name} from {fetched_model_paths[:index]}.")
        else:
            model = fetched_models[0]
            path = fetched_model_paths[0]
            print(f"Using {model_name} from {fetched_model_paths[0]}.")
        return (model, path) if require_model_path else model

    def to(self, device):
        for model in self.model:
            model.to(device)
