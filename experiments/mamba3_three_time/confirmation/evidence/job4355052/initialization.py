"""No forward/fit/evaluation: replay setup and verify five paired initial states."""
import gc
import os
import traceback
import torch
from .config import LOGS, INIT, MODES, SEEDS, pilot
from .provenance import identity, require_evidence, create_record, atomic_json, now
from .config import BATCH
from .runner import prepare
from .state import paired


def main():
    base = identity()
    require_evidence(BATCH, base)
    row = dict(**base, status='RUNNING', rows=[], forward_calls=0, scientific_fits=0)
    create_record(INIT, row)
    try:
        for seed in (2026, *SEEDS):
            pair = []
            for mode in MODES:
                directory = LOGS / 'initialization' / f'{mode}_{seed}'
                directory.mkdir(parents=True, exist_ok=False)
                os.chdir(directory)
                p = dict(runtime=directory, result=directory/'setup.json', checkpoint=directory/'checkpoints/unused.pth',
                         metadata=directory/'checkpoints/unused.json')
                record = dict(**base, mode=mode, seed=seed, run_id=f'initialization_{mode}_{seed}')
                trainer, train, valid, handle = prepare(mode, seed, record, p)
                try:
                    if seed == 2026:
                        original = pilot(mode)
                        for k in ('initial_backbone_sha256', 'initial_calibrator_hashes', 'rng_before_fit_sha256'):
                            if record[k] != original[k]:
                                raise ValueError('Pilot2026 initial procedure drift: ' + k)
                    pair.append(record)
                    row['rows'].append(record)
                    atomic_json(INIT, row)
                finally:
                    handle.remove()
                    trainer.tensorboard.close()
                    del trainer, train, valid, handle
                    gc.collect()
                    torch.cuda.empty_cache()
            paired(*pair)
        row['status'] = 'PASS'
    except BaseException as exc:
        row.update(status='FAIL', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        row['finished_at'] = now()
        atomic_json(INIT, row)


if __name__ == '__main__':
    from experiments.mamba3_three_time.backends import selected
    with selected('upstream'):
        main()
