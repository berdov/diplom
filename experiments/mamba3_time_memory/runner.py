"""One fresh fit with the pilot setup order, trainer and selection semantics."""
import argparse
import hashlib
import json
import os
import signal
import traceback
import torch
from . import config as c
from .progress import write as write_progress
from experiments.mamba3_mimo_time.records import create, update, read, sha, now
from .provenance import identity, require_stage, runtime, imported_sources
from .state import effective_check, initial, paired, require_initial
from experiments.mamba3_mimo_time.provenance import assert_upstream


def train(variant, record, paths):
    from recbole.config import Config
    from recbole.utils import init_seed, init_logger
    from recbole.data.dataloader import TrainDataLoader, FullSortEvalDataLoader
    from recbole.trainer import Trainer
    from experiments.mamba3_timeaware.dataset import PreciseHistoryDataset
    from experiments.mamba3_timeaware.run import verify_protocol, verify_history_stats
    from experiments.mamba3_time_confirmation.config import STATS, STATS_SHA, MANIFEST_SHA
    from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
    from experiments.mamba3_three_time.confirmation.state import precision, canonical_optimizer_settings
    from .trainer import trainer_class
    from .model import TimeMemoryMamba3Rec

    record['runtime'] = runtime(True)
    values = c.settings(variant)
    cfg = Config(model=ThreeTimeMamba3Rec,config_dict=values)
    record.update(effective_config_parity=effective_check(cfg,variant),config=values,
                  effective_config=dict(cfg.final_config_dict),
                  config_sha256=hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest())
    init_seed(cfg['seed']+cfg['local_rank'],cfg['reproducibility'])
    init_logger(cfg)
    protocol = verify_protocol(cfg,check_sha=True)
    dataset = PreciseHistoryDataset(cfg)
    train_ds, valid_ds, reserved = dataset.build()
    del reserved
    if (len(train_ds),len(valid_ds),train_ds.item_num) != (1062567,23951,7112):
        raise ValueError('Frozen data counts changed')
    stats = verify_history_stats(train_ds,read(STATS),cfg)
    train_data = TrainDataLoader(cfg,train_ds,None,shuffle=cfg['shuffle'])
    valid_data = FullSortEvalDataLoader(cfg,valid_ds,None,shuffle=False)
    model = TimeMemoryMamba3Rec(cfg,train_ds).to(cfg['device'])
    assert_upstream(model)
    record.update(protocol=protocol,manifest_sha256=MANIFEST_SHA,train_time_stats_sha256=STATS_SHA,
                  verified_history_stats=stats,parameter_count=sum(p.numel() for p in model.parameters()),history=[],actual_epochs=0)
    if record['parameter_count'] != c.COUNTS[variant]:
        raise ValueError('Parameter count drift')
    previous = [read(c.paths(m)['result']) for m in c.MODES[:c.MODES.index(variant)]]
    base = identity()
    for old in previous:
        if old['status'] != 'PASS' or old['seed'] != 2026 or any(old.get(k) != v for k,v in base.items()):
            raise ValueError('Earlier same-seed mode not complete/current allocation')
    cls = trainer_class(Trainer,valid_data,record,paths,previous[0]['first_train_batch_sha256'] if previous else None)
    trainer = cls(cfg,model)
    trainer.saved_model_file = str(paths['checkpoint'])
    record.update(initial(model,train_data))
    require_initial(record)
    record.update(precision=precision(),optimizer_settings=canonical_optimizer_settings([
        {k:v for k,v in g.items() if k!='params'} for g in trainer.optimizer.param_groups]),
        checkpoint_path=str(paths['checkpoint']),checkpoint_metadata_path=str(paths['metadata']),
        tensorboard_path=str(trainer.tensorboard.log_dir),imported_sources=imported_sources())
    guard = model.times.register_forward_pre_hook(lambda _m,_a:assert_upstream(model))
    try:
        pilot = read(c.PILOT)
        for key in ('protocol','manifest_sha256','train_time_stats_sha256','verified_history_stats','precision','optimizer_settings'):
            if record[key] != pilot[key]:
                raise ValueError('Frozen pilot setting/data drift: '+key)
        for old in previous:
            paired(old,record)
        require_stage(c.SMOKE,base)
        record.update(stage='TRAIN_VALID',scientific_fit_started=True)
        update(paths['result'],record);write_progress(record,paths)
        score, metrics = trainer.fit(train_data,valid_data,saved=True,show_progress=False)
        if float(score) != record['best_valid_score'] or dict(metrics) != record['best_valid_metrics']:
            raise ValueError('Trainer/checkpoint selection mismatch')
        h = record['history']
        if ([r['epoch'] for r in h] != list(range(record['actual_epochs'])) or len(trainer.train_loss_dict) != len(h)
                or sha(paths['checkpoint']) != record['checkpoint_sha256']):
            raise ValueError('History/checkpoint integrity')
        for old in previous:
            paired(old,record,first_batch=True)
        record.update(status='PASS',stage='COMPLETED',first27_best_ndcg10=max(r['valid_ndcg10'] for r in h[:27]) if len(h)>=27 else None,
                      first27_complete=len(h)>=27,train_seconds=sum(r['train_seconds'] for r in h),
                      valid_seconds=sum(r['valid_seconds'] for r in h),
                      peak_gpu_allocated_bytes=max(max(r['train_peak_allocated_bytes'],r['valid_peak_allocated_bytes']) for r in h),
                      peak_gpu_reserved_bytes=max(max(r['train_peak_reserved_bytes'],r['valid_peak_reserved_bytes']) for r in h))
    finally:
        guard.remove()
        trainer.tensorboard.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=c.MODES,required=True)
    args=parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    base=identity()
    if args.variant!='no_memory':
        from .report import validate_record,validate_checkpoint
        previous=read(c.paths('no_memory')['result'])
        validate_record(previous,'no_memory');validate_checkpoint(previous,'no_memory')
        if previous['status']!='PASS':raise ValueError('Fresh no-memory control must pass first')
    gate=require_stage(c.GATE,base)
    smoke=require_stage(c.SMOKE,base)
    p=c.paths(args.variant)
    if any(p[k].exists() or p[k].is_symlink() for k in ('result','lock','checkpoint','metadata')):
        raise FileExistsError('Run already owned; no retry')
    for variant in c.MODES[:c.MODES.index(args.variant)]:
        old=read(c.paths(variant)['result'])
        if old['status']!='PASS' or any(old.get(k)!=v for k,v in base.items()):
            raise ValueError('Earlier planned fit missing/foreign/failed')
    record=dict(**base,mode='dual',memory_mode=args.variant,seed=2026,run_id=p['run_id'],
                targeted_gate_sha256=gate,smoke_sha256=smoke,status='RUNNING',stage='SETUP',
                scientific_fit_started=False,history=[],actual_epochs=0,started_at=now(),
                epoch_indexing='zero-based',selection_split='VALID',evaluation_mode='full-ranking',
                history_length=50,kernel_length_for_max_history=56,
                timing_caveat='Includes first-use JIT/cache effects, not warm-kernel latency')
    create(p['lock'],record);create(p['result'],record)
    def terminate(signum,frame):raise TimeoutError(f'Allocation interrupted: {signum}')
    signal.signal(signal.SIGTERM,terminate)
    progress_initialized=False
    try:
        os.chdir(p['runtime'])
        # Establish the owned status file before setup can reach any fit flag.
        write_progress(record,p);progress_initialized=True
        train(args.variant,record,p)
    except BaseException as exc:
        record.update(status='BLOCKED_OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'INCOMPLETE' if isinstance(exc,TimeoutError) else 'FAIL',
                      error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        record['finished_at']=now();update(p['result'],record)
        if progress_initialized:write_progress(record,p)


if __name__=='__main__':
    main()
