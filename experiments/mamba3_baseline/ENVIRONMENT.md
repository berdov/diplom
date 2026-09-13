# Mamba-3 environment on cHARISMa

Use a **separate environment** from the historical TiM4Rec environment. Do not upgrade `/home/daryumin/iberdov/diplom/envs/tim4rec`.

Target path:

`/home/daryumin/iberdov/diplom/envs/mamba3`

## Why a new environment

The historical TiM4Rec environment uses `mamba-ssm==2.2.2`. Mamba-3 is available in the current official `state-spaces/mamba` source tree and its current package metadata requires newer Mamba-3 dependencies including Triton >= 3.5, TileLang, TVM FFI and quack-kernels. Mutating the old environment would damage reproducibility of the completed TiM4Rec/MTL/MOO work.

## Resolved cHARISMa stack

The environment validated on an NVIDIA A100-SXM4-80GB is:

- Python 3.10.21
- PyTorch 2.9.1+cu128
- torch CUDA runtime 12.8
- RecBole 1.2.0
- Mamba source at `state-spaces/mamba@e9594ce1c732d97440f0332fdc43170a2294dbfa`
- `mamba-ssm==2.3.2.post1`
- Triton 3.5.1
- TileLang 0.1.8
- `apache-tvm-ffi==0.1.9`
- NumPy 1.26.4

The cluster probe used driver 590.48.01, reporting CUDA compatibility 13.1.

## Pinned upstream

Install the official source at:

`state-spaces/mamba@e9594ce1c732d97440f0332fdc43170a2294dbfa`

Do not silently replace this with a PyPI-only `mamba-ssm` install: the experiment is intended to use the Mamba-3 source implementation.

## Minimal environment procedure

Create a separate Python 3.10 environment at `/home/daryumin/iberdov/diplom/envs/mamba3`, then install CUDA-enabled PyTorch, RecBole 1.2.0 and the scientific dependencies used by this baseline.

The validated PyTorch build is:

```bash
PYTHONNOUSERSITE=1 python -m pip install \
  torch==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

Install the official Mamba source:

```bash
PYTHONNOUSERSITE=1 python -m pip install \
  "git+https://github.com/state-spaces/mamba.git@e9594ce1c732d97440f0332fdc43170a2294dbfa" \
  --no-build-isolation
```

At this pinned Mamba commit, the package metadata permits `apache-tvm-ffi<=0.1.12`, but `tilelang==0.1.8` is incompatible with the resolver-selected 0.1.12 on this stack. The observed failure is:

`AttributeError: attribute '__dict__' of 'type' objects is not writable`

Use the working compatibility pin:

```bash
PYTHONNOUSERSITE=1 python -m pip install --force-reinstall \
  "apache-tvm-ffi==0.1.9" \
  "numpy==1.26.4"
```

This pin was validated by both package import and an actual A100 Mamba-3 SISO forward/backward run. The raw check produced finite outputs and finite gradients.

Before the recommender experiment, verify on an allocated A100:

```bash
python - <<'PY'
import torch
import mamba_ssm
from mamba_ssm import Mamba3

print("torch", torch.__version__)
print("cuda", torch.version.cuda, torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0))
print("mamba_ssm", mamba_ssm.__version__)

x = torch.randn(2, 50, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
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
assert torch.isfinite(y).all()
loss = y.float().square().mean()
loss.backward()
assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
print(x.shape, y.shape, True)
PY
```

Record the resolved environment (`python`, PyTorch, CUDA, `mamba-ssm`, Triton, TileLang, TVM FFI and NumPy) in the smoke result. `run.py` does this automatically.

Do not enable MIMO for the first baseline. MIMO adds TileLang-kernel-specific complexity and is a separate architecture choice, not needed to establish the initial Mamba-3 reference.
