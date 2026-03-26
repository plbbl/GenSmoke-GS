# Weights (not shipped in-repo)

| Component | Put file here |
|-----------|----------------|
| Depth Anything V2 ViT-L | `code/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth` ([HF](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true)) |
| UDPNet (ITS) | `code/UDPNet/UDPNet_checkpoints/ConvIR_UDPNet_ITS.ckpt` (and/or `FSNet_UDPNet_ITS.ckpt`) — [UDPNet repo](https://github.com/Harbinzzy/UDPNet) |

Use `code/Depth-Anything-V2/run_smoke_test_depth.py` to build `depth2l/`, then root `run_udp_then_dcp.py`.
