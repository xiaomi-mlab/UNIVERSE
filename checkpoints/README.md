# Checkpoints

Put the released UNIVERSE full checkpoint here:

```text
checkpoints/UNIVERSE.safetensors
```

Inference loads only the `--full_ckpt` file. EMA setup, optimizer state, warmup
configuration, and training-only noise or loss settings are not read by the
released inference entry points.
