"""One immutable validation-only factorial cell or all-four smoke."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import importlib.metadata
import os
import platform
import socket
import time
import traceback
from .common import HERE,ROOT,TARGETS,METRICS,config,combinations,weights,sha,read,write,provenance,reserve,source_digest
from .safety import DataAccessGuard
from .data_audit import summary_checks

def now(): return datetime.now(timezone.utc).isoformat()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--index',type=int,choices=range(16));group.add_argument('--smoke',action='store_true')
    args=p.parse_args();cfg=config();commit=provenance()
    audit=read(HERE/'artifacts/data_audit.json')
    assert audit['status']=='passed' and audit['source_digest']==source_digest()
    if not args.smoke:
        smoke=read(HERE/'runs'/f"{cfg['smoke_run_id']}.json")
        assert smoke['status']=='completed' and smoke['git_commit']==commit and smoke['gates']['passed']
        assert smoke['test_evaluation_count']==0
    cell=deepcopy(combinations()[15 if args.smoke else args.index])
    if args.smoke: cell['run_id']=cfg['smoke_run_id']
    output,artifact=reserve(cell['run_id']);started=time.monotonic()
    payload=dict(cell,experiment_name=cfg['experiment_name'],scientific_run=not args.smoke,status='running',git_commit=commit,git_branch=cfg['branch'],base_main_commit=cfg['base_main_commit'],dataset='KuaiRand-Pure',protocol='B',fingerprint=cfg['fingerprint'],evaluation_split='validation',test_usage='forbidden',test_evaluation_count=0,loss_formula=cfg['loss_formula'],loss_weight_mode=cfg['loss_weight_mode'],lambda_aux=cfg['optimization']['lambda_aux'],individual_aux_weights=weights(cell['active_targets']),auxiliary_loss_weights=weights(cell['active_targets']),effective_pos_weights={t:cfg['optimization']['effective_pos_weights'][t] for t in cell['active_targets']},seed=cfg['training']['seed'],train_batch_size=cfg['training']['train_batch_size'],eval_batch_size=cfg['training']['eval_batch_size'],optimization=cfg['optimization'],max_epochs=1 if args.smoke else cfg['training']['max_epochs'],early_stopping_patience=cfg['training']['early_stopping_patience'],config_sha256=sha(HERE/'config.yaml'),combinations_sha256=sha(HERE/'combinations.yaml'),source_digest=source_digest(),data_audit_sha256=sha(HERE/'artifacts/data_audit.json'),start_time=now(),hostname=socket.gethostname(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),slurm_array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),history=[])
    guard=DataAccessGuard(['/home/daryumin/iberdov/diplom/data',cfg['source']['validation_only_recbole_dir']]);guard.install()
    try:
        execute(payload,cfg,artifact,args.smoke,guard)
        payload['status']='completed'
    except BaseException as exc:
        payload['status']='failed';payload['failure']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        raise
    finally:
        payload.update(end_time=now(),runtime_sec=time.monotonic()-started,data_access=guard.record())
        write(output,payload)

def execute(payload,cfg,artifact,smoke,guard):
    import torch
    from experiments.stage3_auxiliary_analysis import run as old
    import recbole.data.utils as data_utils
    from experiments.multitask_tim4rec_optuna import optuna_search as helpers
    assert torch.cuda.is_available(),'CUDA required; no training on login nodes'
    torch.cuda.reset_peak_memory_stats()
    payload['environment']={'python':platform.python_version(),**{k:importlib.metadata.version(k) for k in ['torch','recbole','mamba_ssm','numpy']},'cuda':torch.version.cuda}
    payload['cuda_device']=str(torch.cuda.current_device());payload['gpu_name']=torch.cuda.get_device_name()
    def forbidden(*a,**kw): raise RuntimeError('Optuna studies forbidden')
    helpers.optuna.create_study=forbidden;helpers.optuna.load_study=forbidden
    old.get_dataloader=guard.loader_factory(old.get_dataloader)
    data_utils.get_dataloader=guard.loader_factory(data_utils.get_dataloader)
    stage=deepcopy(cfg);stage['outputs']['artifact_dir']=str(artifact)
    opt=old.load_yaml(stage['source']['validation_only_config'])
    old.assert_protocol_config(opt)
    summary=old.load_json(opt['validation_only_data']['summary_json'])
    old.assert_validation_only_summary(summary);summary_checks(summary)
    data=old.load_data_bundle(stage)
    run_cfg={'run_id':payload['run_id'],'active_targets':payload['active_targets']}
    rc=old.build_run_recbole_config(stage,data.base_config,run_cfg,1 if smoke else None)
    assert rc['benchmark_filename']==['train','valid'] and rc['eval_args']['split']=={'LS':'valid_only'}
    assert rc['metrics']==['Hit','Recall','NDCG'] and list(rc['topk'])==[5,10,20,50]
    assert rc['MAX_ITEM_LIST_LENGTH']==50 and rc['final_test_evaluation_count']==0
    old.init_seed(rc['seed']+rc['local_rank'],rc['reproducibility'])
    train,valid=old.create_loaders(rc,data.train_dataset,data.valid_dataset)
    # Check loader semantics without changing the RNG trajectory of model creation.
    rng=old.rng_snapshot()
    for batch in valid: assert batch[1] is None,'Seen-item masking is forbidden for sequential full ranking'
    old.restore_rng(rng)
    model=old.MultitaskTiM4Rec(rc,train.dataset).to(rc['device'])
    assert tuple(model.input_fields_used)==('item_id_list','item_length','timestamp_list')
    active=payload['active_targets'];op=cfg['optimization']
    optimizer=old.optimizer_for_model(model,active,op['learning_rate'],op['head_learning_rate'],op['weight_decay'])
    # Hook only checks gradients; no new clipping/scaling/update rule.
    def check_gradients(opt,args,kwargs):
        for g in opt.param_groups:
            for param in g['params']:
                if param.grad is not None and not torch.isfinite(param.grad).all(): raise RuntimeError('Nonfinite gradient')
    optimizer.register_step_pre_hook(check_gradients)
    trainer=old.Trainer(rc,model);trainer.optimizer=optimizer
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    best=-float('inf');bad=0;records=[];train_seconds=0
    for epoch in range(1,payload['max_epochs']+1):
        start=time.monotonic()
        stats,diag=old.train_one_epoch(model,train,optimizer,active,op['lambda_aux'],weights(active),payload['effective_pos_weights'],epoch,cfg['training']['gradient_diagnostics'],max_train_batches=cfg['smoke_batches'] if smoke else None)
        train_seconds+=time.monotonic()-start;records.extend(diag)
        for param in model.parameters(): assert torch.isfinite(param).all(),'Nonfinite parameter'
        valid_result,checks=old.evaluate_full_sort_with_checks(trainer,valid,train)
        old.check_hit_recall_equal(valid_result,list(rc['topk']))
        assert checks['raw_scores_all_finite'] and checks['positive_scores_all_finite'] and checks['candidate_universe_size']==7111
        metrics=old.normalize_metric_keys(old.metric_subset(valid_result));assert set(metrics)==set(METRICS)
        score=metrics['NDCG@10'];improved=score>best
        if improved:
            best=score;bad=0
            payload.update(best_epoch=epoch,best_validation_ndcg10=score,ranking_metrics=metrics)
            checkpoint=artifact/'best_validation.pth';tmp=artifact/'best_validation.tmp'
            torch.save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'metrics':metrics,'git_commit':payload['git_commit'],'run_id':payload['run_id'],'test_evaluation_count':0},tmp);tmp.replace(checkpoint)
            payload['best_checkpoint']=str(checkpoint)
        else: bad+=1
        payload['history'].append({'epoch':epoch,'train':stats,'ranking_metrics':metrics,'full_ranking_checks':checks,'is_best':improved})
        payload.update(actual_epochs=epoch,stopped_early=bad>=payload['early_stopping_patience'],training_time_sec=train_seconds)
        write(artifact/'progress.json',payload)
        print(f"{payload['run_id']} epoch={epoch} NDCG@10={score} best={best}",flush=True)
        if payload['stopped_early']: break
    deltas={n:float(torch.linalg.vector_norm(p.detach()-before[n])) for n,p in model.named_parameters()}
    for t in TARGETS:
        changed=any(d>0 for n,d in deltas.items() if n.startswith(old.head_prefix(t)))
        assert changed==(t in active),f'Unexpected update for head {t}'
    assert any(d>0 for n,d in deltas.items() if not any(n.startswith(old.head_prefix(t)) for t in TARGETS))
    assert not guard.blocked_test_accesses
    payload['gradient_diagnostics']={'policy':cfg['training']['gradient_diagnostics'],'raw_records':old.json_ready(records),'summary':old.json_ready(old.aggregate_gradient_records(records))}
    payload['gates']={'passed':True,'active_heads_updated_only':True,'backbone_updated':True,'finite':True,'validation_checked':True,'test_evaluation_count':0}
    payload['peak_gpu_memory_bytes']=torch.cuda.max_memory_allocated()
    payload['loader_inspection']=data.loader_inspection

if __name__=='__main__': main()
