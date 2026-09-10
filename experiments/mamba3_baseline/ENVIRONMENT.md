# Mamba-3 environment on cHARISMa

Use a **separate environment** from the historical TiM4Rec environment. Do not upgrade `/home/daryumin/iberdov/diplom/envs/tim4rec`.

Target path:

`/home/daryumin/iberdov/diplom/envs/mamba3`

## Why a new environment

The historical TiM4Rec environment uses `mamba-ssm==2.2.2`. Mamba-3 is available in the current official `state-spaces/mamba` source tree and its current package metadata requires newer Mamba-3 dependencies including Triton >= 3.5, TileLang, TVM FFI and quack-kernels. Mutating the old environment would damage reproducibility of the completed TiM4Rec/MTL/MOO work.

## Pinned upstream

Install the official source at:

`state-spaces/mamba@e9594ce1c732d97440f0332fdc43170a2294dbfa`

Do not silently replace this with a PyPI-only `mamba-ssm` install: the experiment is intended to use the Mamba-3 source implementation.

## Minimal environment procedure

First inspect the compute-node NVIDIA driver / CUDA compatibility and choose a CUDA-enabled PyTorch build supported by that driver. Then, in a fresh Python >= 3.10 conda environment, install:

- CUDA-enabled PyTorch
- RecBole 1.2.0 and the same data/scientific dependencies used by the TiM4Rec reproduction
- official Mamba source pinned to the commit above with `--no-build-isolation`

Source install pattern:

```bash
PYTHONNOUSERSITE=1 pip install \
  "git+https://github.com/state-spaces/mamba.git@e9594ce1c732d97440f0332fdc43170a2294dbfa" \
  --no-build-isolation
```

The official source currently declares, among others, `triton>=3.5.0`, `tilelang==0.1.8`, `apache-tvm-ffi<=0.1.12`, and `quack-kernels>=0.3.4`.

Before the experiment, verify:

```bash
python - <<'PY'
import torch
import mamba_ssm
from mamba_ssm import Mamba3

print("torch", torch.__version__)
print("cuda", torch.version.cuda, torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0))
print("mamba_ssm", mamba_ssm.__version__)

x = torch.randn(2, 50, 64, device="cuda", dtype=torch.bfloat16)
m = Mamba3(
    d_model=64,
    d_state=128,
    expand=2,
    headdim=64,
    ngroups=1,
    rope_fraction=0.5,
    chunk_size=64,
    is_mimo=False,
    is_outproj_norm=False,
    dtype=torch.bfloat16,
).cuda()
y = m(x)
print(x.shape, y.shape, torch.isfinite(y).all().item())
PY
```

Record the resolved environment (`python`, PyTorch, CUDA, `mamba-ssm`, Triton, TileLang) in the smoke result. `run.py` does this automatically.

Do not enable MIMO for the first baseline. MIMO adds TileLang-kernel-specific complexity and is a separate architecture choice, not needed to establish the initial Mamba-3 reference.
