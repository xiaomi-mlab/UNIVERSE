<div align="center">

# UNIVERSE

### Unified Video Action Models for Autonomous Driving with Flexible Mask-Modulated Modality Generation

**NeurIPS 2026**

Mengmeng Liu<sup>1</sup>, Diankun Zhang<sup>2</sup>, Jiuming Liu<sup>3</sup>,<br>
Jianfeng Cui<sup>2</sup>, Hongwei Xie<sup>2</sup>, Guang Chen<sup>2</sup>,
Hangjun Ye<sup>2</sup>,<br>
Michael Ying Yang<sup>4</sup>, Francesco Nex<sup>1</sup>, Hao Cheng<sup>1</sup>

<sup>1</sup>University of Twente&nbsp;&nbsp;
<sup>2</sup>Xiaomi EV&nbsp;&nbsp;
<sup>3</sup>University of Cambridge&nbsp;&nbsp;
<sup>4</sup>University of Bath

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2607.05133)
![Code](https://img.shields.io/badge/Code-Pending%20Approval-f0ad4e.svg)

</div>

> **Code release status:** The training and inference code is undergoing final
> approval and will be added to this repository after approval. This repository
> currently provides the project documentation only.

## News

- **September 25th, 2026:** UNIVERSE is accepted by NeurIPS 2026.
- **July 6th, 2026:** UNIVERSE is released on [arXiv](https://arxiv.org/abs/2607.05133).

## Overview

UNIVERSE is a unified video-action model based on a single mask-modulated
Diffusion Transformer. Future video latents and ego-trajectory tokens are
co-trained with shared parameters, while the Modality-Decoupling Visibility
Mask prevents leakage between future modalities. The same model supports
video-only, trajectory-only, and joint video-trajectory generation.

<div align="center">
<b>UNIVERSE shares video and trajectory denoising in a single DiT.</b>
<br><br>
<img src="https://arxiv.org/html/2607.05133v1/first.png" alt="Comparison between UNIVERSE and prior video-action architectures" />

<br><br>
<b>Overall pipeline and flexible inference modes.</b>
<br><br>
<img src="https://arxiv.org/html/2607.05133v1/pipeline.png" alt="UNIVERSE architecture and inference modes" />
</div>

The visibility mask shares historical context across modalities but blocks
attention between future video and trajectory targets. At inference time, the
future token groups determine whether the model produces video, trajectory, or
both. Trajectory-only inference removes future-video denoising and provides a
4.3x speedup over joint video-action rollout while maintaining comparable
planning accuracy.

## Qualitative Results

<div align="center">
<b>Zero-shot transfer from NavSIM to nuScenes and Bench2Drive.</b>
<br><br>
<img src="https://arxiv.org/html/2607.05133v1/visual2.png" alt="UNIVERSE zero-shot predictions on nuScenes and Bench2Drive" />
</div>

UNIVERSE is trained on NavSIM and evaluated on nuScenes and Bench2Drive without
target-dataset fine-tuning. See the [paper](https://arxiv.org/abs/2607.05133)
for the complete quantitative results, ablation studies, and evaluation
protocol.

## Code Release

After approval, this repository will provide:

- NavSIM v1 training code.
- NavSIM, nuScenes, and Bench2Drive inference entry points.
- Configuration files and environment-variable path overrides.
- Reproducible evaluation instructions.

## Acknowledgments

UNIVERSE builds on and is inspired by the following open-source projects and
benchmarks:

- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- [Wan2.2](https://github.com/Wan-Video/Wan2.2)
- [NAVSIM](https://github.com/autonomousvision/navsim)
- [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit)
- [nuPlan devkit](https://github.com/motional/nuplan-devkit)
- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [Diffusers](https://github.com/huggingface/diffusers)

## Citation

```bibtex
@article{liu2026universe,
  title={UNIVERSE: Unified Video Action Models for Autonomous Driving with Flexible Mask-Modulated Modality Generation},
  author={Liu, Mengmeng and Zhang, Diankun and Liu, Jiuming and Cui, Jianfeng and Xie, Hongwei and Chen, Guang and Ye, Hangjun and Nex, Francesco and Cheng, Hao and Yang, Michael Ying},
  journal={arXiv preprint arXiv:2607.05133},
  year={2026}
}
```
