# 2D Enhancement

**Order:** UDPNet → DCP → **GPT-Image-1.5** (required, **one image per API call**) → write MLLM outputs into each scene’s **`train/`**, then run [`../3d_reconstruction/train.sh`](../3d_reconstruction/train.sh) on that dataset parent.

Overview: [`../README.md`](../README.md).

## Install

```bash
pip install -r requirements.txt -r dcp_dehaze/requirements.txt
```

## UDP + DCP

```bash
python run_udp_then_dcp.py --scene_roots <develop_parent> <test_parent> --out_root <udp_dcp_out> --ckpt code/UDPNet/UDPNet_checkpoints/ConvIR_UDPNet_ITS.ckpt
```

Weights / `depth2l`: [`WEIGHTS_README.md`](WEIGHTS_README.md).

## MLLM (GPT-Image-1.5)

- Model: **`gpt-image-1.5`** (OpenAI API). Docs: [image generation](https://platform.openai.com/docs/guides/image-generation), [model](https://platform.openai.com/docs/models/gpt-image-1.5), [API ref](https://platform.openai.com/docs/api-reference/images).
- Default prompt (most scenes; a few scenes tweaked): [`gpt_image_prompt_default.txt`](gpt_image_prompt_default.txt).
- Example outputs (Baidu): [link](https://pan.baidu.com/s/1M14Tw5RY42ovroslUz0PWA) (code on share page).
