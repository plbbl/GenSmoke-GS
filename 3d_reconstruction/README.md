# 3D Reconstruction

**Input:** Each scene directory must have **`train/`** with **post-MLLM** views. Run the 2D stages first: [`../2d_enhancement/README.md`](../2d_enhancement/README.md) (UDPNet → DCP → GPT-Image-1.5).

**This directory:** 3DGS-MCMC + FasterGS training and multi-run NVS averaging — see `train.sh` and `scripts/`.

Overview: [`../README.md`](../README.md).

## Install

```bash
bash install.sh && sudo apt install -y colmap
```

## Train

```bash
./train.sh /path/to/dataset_parent
```
