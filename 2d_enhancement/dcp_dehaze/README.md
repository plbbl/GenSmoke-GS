# DCP batch dehazing

Dark channel prior (He et al., CVPR 2009) with guided-filter transmission refinement. Default hyperparameters are fixed in `params.json` (edit this file to change them). Command-line flags override individual values.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python batch_dehaze.py --input /path/to/images --output /path/to/out
```

Recursively reads common image extensions and mirrors the directory layout under `--output`.

## Third-party code

`dark-channel-prior-dehazing/src/dehaze.py` follows the reference implementation at [joyeecheung/dark-channel-prior-dehazing](https://github.com/joyeecheung/dark-channel-prior-dehazing). `guidedfilter_fast.py` is a faster guided-filter variant (box-filter form, luminance guidance) used for large batches.
