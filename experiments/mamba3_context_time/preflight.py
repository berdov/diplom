"""Login checks are imports/config/source hashes/CPU construction, not dataset passes."""
import importlib.metadata
import importlib.util
import json

from .config import ROOT, plan, settings
from .provenance import PIN, verify, backbone_hash, rng_hash, tensor_hash


def effective_settings_check(config, task):
    from recbole.config import Config
    from experiments.mamba3_time_mechanisms.model import MechanismMamba3Rec
    frozen = settings(task)
    frozen.pop('context_time_mode')
    reference = Config(model=MechanismMamba3Rec, config_dict=frozen)
    ignored = {'model', 'context_time_mode'}
    a = {k: v for k,v in config.final_config_dict.items() if k not in ignored}
    b = {k: v for k,v in reference.final_config_dict.items() if k not in ignored}
    if a != b:
        different = [k for k in set(a)|set(b) if a.get(k) != b.get(k)]
        raise ValueError(f'Effective frozen configuration drift: {sorted(different)}')
    return 'PASS (except model name and context_time_mode)'


def runtime_check(require_cuda=False):
    import torch
    reference = json.loads((ROOT / plan()['historical_reference']['source_json']).read_text())
    names = {'torch': 'torch', 'recbole': 'recbole', 'mamba_ssm': 'mamba-ssm', 'triton': 'triton',
             'numpy': 'numpy', 'tilelang': 'tilelang', 'apache_tvm_ffi': 'apache-tvm-ffi'}
    versions = {key: importlib.metadata.version(dist) for key, dist in names.items()}
    for key, actual in versions.items():
        if actual != reference['runtime'][key]:
            raise ValueError(f'Frozen library drift: {key}: {actual}')
    direct = json.loads(importlib.metadata.distribution('mamba-ssm').read_text('direct_url.json'))
    if direct['vcs_info']['commit_id'] != PIN:
        raise ValueError('Pinned Mamba commit mismatch')
    if require_cuda and (not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name()):
        raise RuntimeError('Real A100 CUDA required; no CPU fallback')
    versions.update(pinned_mamba=PIN, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name() if require_cuda else None)
    return versions


def full_counts():
    from recbole.config import Config
    from recbole.utils import init_seed
    from .model import ContextMamba3Rec
    from .gpu_checks import Catalog
    from .temporal import COUNTS, TEMPORAL_COUNTS
    rows = []
    for task in plan()['tasks']:
        cfg = Config(model=ContextMamba3Rec, config_dict=settings(task))
        effective_settings_check(cfg, task)
        init_seed(2026, True)
        model = ContextMamba3Rec(cfg, Catalog())
        count = sum(p.numel() for p in model.parameters())
        temporal = sum(p.numel() for p in model.mechanisms.parameters())
        if (count, temporal) != (COUNTS[task['mode']], TEMPORAL_COUNTS[task['mode']]):
            raise ValueError('Actual count mismatch')
        rows.append(dict(mode=task['mode'], total=count, temporal=temporal,
            backbone=backbone_hash(model), rng=rng_hash(),
            bank=tensor_hash(model.mechanisms.bank.state_dict()) if task['mode'] in ('uniform','routed') else None))
    if len({r['backbone'] for r in rows}) != 1 or len({r['rng'] for r in rows}) != 1 or rows[3]['bank'] != rows[4]['bank']:
        raise ValueError('CPU full-model initialization parity failed')
    return rows


def main():
    sources = verify()
    runtime = runtime_check()
    if importlib.util.find_spec('matplotlib') is None:
        raise ImportError('Existing matplotlib required for the predeclared plot; no installation permitted')
    rows = full_counts()
    # Stat only; full data SHA and reference-stat checks belong on the compute node.
    path = ROOT / 'data/processed/protocol_b/recbole/kuairand/kuairand.inter'
    if not path.is_file():
        raise FileNotFoundError(path)
    print(json.dumps(dict(status='PASS', source_hash=sources['source_hash'], runtime=runtime,
                         parameters=rows, dataset_path=str(path), dataset_size=path.stat().st_size), indent=2))


if __name__ == '__main__':
    main()
