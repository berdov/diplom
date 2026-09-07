"""Guarded validation-only stages. No TEST, tuning, overwrite, or resume options."""
import argparse
import importlib.metadata
import json
import os
import platform
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
import yaml
from .common import HERE, ROOT, digest, git, load_config, require_gate, source_digest, verify_historical_inputs, write_json
from .safety import DataAccessGuard, validate_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',required=True,choices=['ferero','most','phn_hvi'])
    parser.add_argument('--stage',required=True,choices=['smoke','sanity','convergence_screening'])
    parser.add_argument('--preflight-only',action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    historical = verify_historical_inputs()
    upstream = yaml.safe_load((HERE/'provenance.yaml').read_text())['methods'][args.method]
    input_config = yaml.safe_load((ROOT/config['source']['optuna_config']).read_text())
    validate_config(config,input_config)
    gate = require_gate(args.method,args.stage)
    sha, branch = git('rev-parse','HEAD'), git('branch','--show-current')
    if branch != config['branch'] or git('status','--porcelain','--untracked-files=no'):
        raise RuntimeError('Expected clean committed challenger branch')
    if os.environ.get('MOO_GIT_COMMIT') != sha:
        raise RuntimeError('MOO_GIT_COMMIT must match checked-out commit exactly')
    # Cluster job must run the SHA already published on this branch.
    if git('rev-parse','origin/'+branch) != sha:
        raise RuntimeError('Challenger HEAD is not the fetched published branch SHA')
    run_id = f"{args.method}_{'convergence' if args.stage=='convergence_screening' else args.stage}_001"
    artifact = HERE/'artifacts'/run_id
    output = HERE/'runs'/f'{run_id}.json'
    if output.exists() or artifact.exists():
        raise RuntimeError(f'Refusing to overwrite or repeat existing run: {run_id}')
    if args.preflight_only:
        print(json.dumps({'run_id':run_id,'git_commit':sha,'status':'preflight_passed','test_evaluation_count':0}))
        return
    artifact.mkdir(parents=True)
    started = time.monotonic()
    payload = dict(run_id=run_id,method=args.method,stage=args.stage,status='running',seed=config['run']['seed'],
        git_commit=sha,git_branch=branch,git_remote=git('remote','get-url','origin'),
        source_digest=source_digest(),protocol_fingerprint=config['protocol'],dataset='KuaiRand-Pure / Protocol B',
        users=config['protocol']['users'],items=config['protocol']['items'],train_rows=config['protocol']['train'],
        validation_rows=config['protocol']['validation'],task_order=config['objectives']['task_order'],
        test_evaluation_count=0,training={},validation=None,auxiliary=None,objective_vector=None,
        method_config=config['methods'][args.method],operating_point_selection=config['selection'],
        frozen_config=config,config_sha256=digest(HERE/'config.yaml'),provenance_sha256=digest(HERE/'provenance.yaml'),
        historical_inputs=historical,entry_gates=gate,started_utc=datetime.now(timezone.utc).isoformat(),
        slurm={k:os.environ.get(k) for k in ['SLURM_JOB_ID','SLURM_JOB_PARTITION','SLURM_JOB_NODELIST','SLURM_RESTART_COUNT']},
        hostname=socket.gethostname(),python=platform.python_version(),**upstream)
    guard = DataAccessGuard([config['protocol']['processed_dir'],config['protocol']['multitask_dir'],
                             input_config['validation_only_data']['output_root']])
    guard.install()
    try:
        execute(args,config,input_config,artifact,payload,guard)
        payload['status']='completed'
    except BaseException as exc:
        payload['status']='failed'
        payload['failure']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        raise
    finally:
        payload['runtime_seconds']=time.monotonic()-started
        payload['finished_utc']=datetime.now(timezone.utc).isoformat()
        payload['data_access']=guard.record()
        write_json(output,payload)
        (HERE/'runs'/f'{run_id}_notes.md').write_text(
            f"# {run_id}\n\nStatus: {payload['status']}. Commit: `{sha}`. TEST evaluations: 0.\n\n"
            f"Implementation: {payload['implementation_name']}; exact reproduction: false.\n\n"
            + '\n'.join('- '+d for d in payload['upstream']['deviations']) + '\n')


def execute(args,config,input_config,artifact,payload,guard):
    import numpy as np
    import torch
    from experiments.moo_8families import train as old
    from experiments.multitask_tim4rec_optuna import optuna_search as data_helpers
    from recbole.data import utils as data_utils
    from .methods.most import MosT
    from .training import train_epoch, evaluate, parameter_hash, parameter_snapshot, update_deltas
    factory = guard.loader_factory(data_utils.get_dataloader)
    data_utils.get_dataloader = factory
    data_helpers.get_dataloader = factory
    # The imported historical module supplies pure helpers only. Disable accidental study creation.
    def no_study(*a,**kw): raise RuntimeError('Optuna execution forbidden for challengers')
    data_helpers.optuna.create_study = no_study
    data_helpers.optuna.load_study = no_study
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required; never train on login node')
    torch.cuda.reset_peak_memory_stats()
    payload['environment']={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    payload['gpu']={'name':torch.cuda.get_device_name(),'total_memory_bytes':torch.cuda.get_device_properties(0).total_memory}
    # Override only disposable model/cache outputs, never prepare/rebuild shared data.
    data = old.load_data_bundle(input_config,artifact/'data_probe')
    locked = old.load_yaml(ROOT/config['source']['best_params'])
    sampled = old.sampled_from_locked_params(locked,data.target_stats)
    for key in ['learning_rate','weight_decay','dropout_prob','head_lr_multiplier','lambda_aux']:
        if sampled[key] != config['tuned_fixed_params'][key]: raise RuntimeError(f'Locked parameter changed: {key}')
    if sampled['normalized_task_weights'] != config['tuned_fixed_params']['normalized_task_weights']:
        raise RuntimeError('Locked auxiliary weights changed')
    epochs = 1 if args.stage=='smoke' else config['run']['sanity_epochs'] if args.stage=='sanity' else config['run']['convergence_max_epochs']
    cfg = old.build_run_config(input_config,artifact,sampled,epochs)
    old.assert_protocol_guards(config,data,cfg)
    old.init_seed(config['run']['seed']+cfg['local_rank'],cfg['reproducibility'])
    train_data,valid_data = old.create_loaders(cfg,data.train_dataset,data.valid_dataset)
    preferences_file = old.load_preferences(ROOT/config['source']['preferences'])
    mcfg = config['methods'][args.method]
    preferences = old.preference_records(preferences_file,mcfg['preference_set']) if args.method!='most' else None
    models=[]
    for i in range(len(preferences) if args.method=='ferero' else mcfg['solution_count'] if args.method=='most' else 1):
        old.init_seed(config['run']['seed']+i,cfg['reproducibility'])
        if args.method=='phn_hvi':
            class ReplaySafePHN(old.PHNAdapterTiM4Rec):
                def preference(self,reference=None):
                    return super().preference(reference).clone()
            model = ReplaySafePHN(cfg,train_data.dataset,adapter_hidden_size=mcfg['adapter_hidden_size'],adapter_scale=mcfg['adapter_scale']).to(cfg['device'])
        else:
            model = old.MultitaskTiM4Rec(cfg,train_data.dataset).to(cfg['device'])
        models.append(model)
    optimizers=[old.optimizer_for_trial(m,sampled) for m in models]
    trainers=[old.Trainer(cfg,m) for m in models] if args.stage!='smoke' else []
    pos_weights=old.pos_weight_tensors(sampled['effective_pos_weights'],cfg['device'])
    normalization=old.compute_normalization_diagnostics(models[0],train_data,sampled=sampled,pos_weights=pos_weights,
        batches=config['normalization']['diagnostic_batches'],selector=config['normalization']['gradient_selector'])
    payload['normalization']=normalization
    payload['loader_inspection']=data.loader_inspection
    payload['sampled_locked_parameters']=sampled
    payload['training']={'epochs':[],'best_epoch':None,'early_stopping':old.early_stopping_config_for_stage(config,args.stage),
                         'learning_rate':sampled['learning_rate'],'optimizer_learning_rates':old.optimizer_learning_rates(optimizers),
                         'parameter_count':[old.count_parameters(m) for m in models],
                         'sequential_train_examples':len(data.train_dataset),'effective_train_batch_size':cfg['train_batch_size']}
    state = MosT(**mcfg['solver']) if args.method=='most' else None
    rng=np.random.default_rng(mcfg.get('preference_seed',config['run']['seed']))
    initial_hashes=[parameter_hash(m) for m in models]
    before=[parameter_snapshot(m) for m in models]
    best=None; best_score=None; best_epoch=None; stale=0; checks=0; stopped=False
    interval=1 if args.stage=='sanity' else config['run']['convergence_validation_interval']
    for epoch in range(1,epochs+1):
        record=train_epoch(args.method,models,optimizers,train_data,sampled,pos_weights,normalization['loss_scales'],
            mcfg,preferences,state,rng,old,max_batches=config['run']['smoke_batches'] if args.stage=='smoke' else None)
        record['epoch']=epoch
        if args.stage!='smoke' and epoch % interval==0:
            validation=evaluate(models,trainers,preferences,args.method,train_data,valid_data,config,old)
            score=float(validation['ranking_operating_point']['metrics']['NDCG@10']); checks+=1
            record['validation']=validation
            if best_score is None or score>best_score+config['run']['convergence_min_delta']:
                best,best_score,best_epoch,stale=validation,score,epoch,0
                payload['best_checkpoint']=old.save_checkpoint(artifact/'checkpoints/best_validation.pth',models,optimizers,epoch,score,
                    {'git_commit':payload['git_commit'],'run_id':payload['run_id'],'test_evaluation_count':0,
                     'selection':config['selection'],'source_digest':payload['source_digest']})
            else: stale+=1
            stopped=(args.stage=='convergence_screening' and epoch>=config['run']['convergence_min_epochs'] and stale>=config['run']['convergence_patience'])
        payload['training']['epochs'].append(record)
        payload['training'].update(best_epoch=best_epoch,validation_checks=checks,checks_without_improvement=stale,
                                  early_stopped=stopped,stop_epoch=epoch,
                                  peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                                  peak_memory_reserved_bytes=torch.cuda.max_memory_reserved())
        payload['last_checkpoint']=old.save_checkpoint(artifact/'checkpoints/last.pth',models,optimizers,epoch,best_score,
            {'git_commit':payload['git_commit'],'run_id':payload['run_id'],'test_evaluation_count':0,
             'normalization':normalization,'method_state':state.__dict__ if state else None,
             'preference_rng_state':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
             'cuda_rng':torch.cuda.get_rng_state_all(),'source_digest':payload['source_digest'],
             'resume_supported':False})
        payload['data_access']=guard.record()
        write_json(artifact/'progress.json',payload)
        print(json.dumps({'run_id':payload['run_id'],'epoch':epoch,'train_seconds':record['runtime_seconds'],
                          'best_epoch':best_epoch,'best_validation_ndcg10':best_score,'test_evaluation_count':0}),flush=True)
        if stopped: break
    final_hashes=[parameter_hash(m) for m in models]
    deltas=[update_deltas(m,b) for m,b in zip(models,before)]
    sensitivity=None
    if args.method=='phn_hvi':
        sensitivity=old.preference_sensitivity_diagnostic(model=models[0],train_data=train_data,preferences=preferences_file,
            p1_id='rank_heavy',p2_id='like_heavy',tolerance=1e-8)
        if not sensitivity['output_metric_passed']: raise RuntimeError('Zero PHN preference sensitivity')
    if any(d.get('backbone',0)<=0 for d in deltas): raise RuntimeError('Backbone parameters did not update')
    for head in ['click_head','long_view_head','like_head','profile_enter_head']:
        if not any(d.get(head,0)>0 for d in deltas): raise RuntimeError(f'Head parameters did not update: {head}')
        if not any(e['gradient_norm_max_by_model'][i].get(head,0)>0 for e in payload['training']['epochs'] for i in range(len(models))):
            raise RuntimeError(f'Head had no nonzero gradients: {head}')
    if args.method=='most' and (len(set(initial_hashes))!=3 or len(set(final_hashes))!=3):
        raise RuntimeError('MosT solutions collapsed to identical parameters')
    if args.method=='ferero':
        first=payload['training']['epochs'][0]['first_method_state']['solutions']
        last=payload['training']['epochs'][-1]['last_method_state']['solutions']
        if all(np.allclose(a['coefficients'],b['coefficients']) for a,b in zip(first,last)):
            raise RuntimeError('FERERO coefficients did not change across batches')
    if guard.blocked_test_accesses: raise RuntimeError('Blocked TEST attempt occurred')
    payload['gates']={'passed':True,'initial_parameter_hashes':initial_hashes,'final_parameter_hashes':final_hashes,
                      'parameter_update_l2':deltas,'preference_sensitivity':sensitivity,
                      'finite_losses_gradients_parameters':True,'test_evaluation_count':0}
    payload['validation']=best
    payload['auxiliary']=best['ranking_operating_point']['auxiliary_validation'] if best else None
    payload['objective_vector']=best['ranking_operating_point']['objective_vector'] if best else None
    payload['training']['runtime']=sum(e['runtime_seconds'] for e in payload['training']['epochs'])
    if best is not None and best_epoch is None: raise RuntimeError('Missing best epoch')


if __name__=='__main__':
    main()
