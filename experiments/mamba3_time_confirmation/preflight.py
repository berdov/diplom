"""Login-node import/CPU-constructor preflight; no data loading or forward."""
import json

from .config import COUNTS, ROOT, paths, plan, settings
from .provenance import verify


def main():
    manifest = verify()
    for task in plan()['tasks']:
        p=paths(task)
        for key in ('result','lock','smoke_checkpoint'):
            if p[key].exists():
                raise FileExistsError(f'No retry/overwrite: {p[key]}')
        if p['checkpoints'].exists() and any(p['checkpoints'].iterdir()):
            raise FileExistsError('Checkpoint directory not empty')
    import torch
    from recbole.config import Config
    from recbole.utils import init_seed
    from .model import model_class
    from .run import backbone_hash
    from experiments.mamba3_timeaware.run import runtime_info
    from experiments.mamba3_time_mechanisms.provenance import PIN
    import importlib.metadata
    installed=json.loads(importlib.metadata.distribution('mamba-ssm').read_text('direct_url.json'))
    if installed['vcs_info']['commit_id']!=PIN:
        raise ValueError('Mamba pin mismatch')

    class SyntheticDataset:
        def num(self, field):
            return 7112 if field=='item_id' else 23952

    models={}
    for mode in COUNTS:
        task=dict(plan()['tasks'][0],mode=mode)
        config=Config(model=model_class(mode),config_dict=settings(task))
        json.dumps(config.final_config_dict,allow_nan=False,default=str)
        init_seed(2027,True)
        model=model_class(mode)(config,SyntheticDataset())  # CPU constructor, no forward.
        if next(model.parameters()).device.type!='cpu':
            raise ValueError('Preflight must stay on CPU')
        count=sum(p.numel() for p in model.parameters())
        if count!=COUNTS[mode]:raise ValueError('Parameter count mismatch')
        models[mode]=dict(parameters=count,backbone_sha256=backbone_hash(model))
        del model
    if len({v['backbone_sha256'] for v in models.values()})!=1:
        raise ValueError('Paired initialization differs')
    runtime=runtime_info()
    reference=json.loads((ROOT/plan()['reuse'][0]['source_json']).read_text())
    for key in ('torch','recbole','mamba_ssm','triton','numpy'):
        if runtime[key]!=reference['runtime'][key]:raise ValueError(f'Runtime drift: {key}')
    print(json.dumps(dict(status='PASS',study_source_hash=manifest['source_hash'],pinned_mamba=PIN,
                         imports='PASS',CPU_constructors=models,forward_calls=0,dataset_loads=0,
                         TRAIN=0,VALID=0,TEST=0),indent=2))


if __name__=='__main__':
    main()
