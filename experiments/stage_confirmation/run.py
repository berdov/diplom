"""Six fixed seed confirmations using the byte-identical screening execute loop."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from experiments.target_combination_analysis import run as frozen
from experiments.target_combination_analysis.common import config, combinations, weights, sha, write
from experiments.target_combination_analysis.safety import DataAccessGuard

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (2026, 2027, 2028)
CELLS = (0, 3, 11)

def case(index):
    if index not in range(6): raise ValueError('Only the six frozen confirmations are authorized')
    seed = SEEDS[1 + index // 3]
    cell = deepcopy(combinations()[CELLS[index % 3]])
    cell['screening_run_id'] = cell['run_id']
    cell['run_id'] = f"target_confirm_{cell['combination_id']}_seed{seed}_001"
    cfg = deepcopy(config())
    cfg['training']['seed'] = seed
    return cell, cfg

def verify_sources():
    hashes = json.loads((HERE/'frozen_source_hashes.json').read_text())
    for path, expected in hashes.items():
        if sha(ROOT/path) != expected: raise RuntimeError(f'Frozen scientific source changed: {path}')
    smoke = json.loads((ROOT/'reports/evidence/target_combinations/runs/target_combo_smoke_all_four_002.json').read_text())
    assert smoke['status']=='completed' and smoke['gates']['passed'] and smoke['test_evaluation_count']==0
    assert smoke['git_commit']=='599bcdb6e50dceea73233169b8834eaceef083a8'
    return len(hashes)

def provenance():
    verify_sources()
    def git(*a): return subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()
    commit=git('rev-parse','HEAD')
    assert os.environ['SC_COMMIT']==commit
    assert git('branch','--show-current')=='codex/close-experimental-stage'
    assert not git('status','--porcelain','--untracked-files=no')
    assert git('rev-parse','origin/codex/close-experimental-stage')==commit
    return commit

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=int,choices=range(6),required=True)
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args();commit=provenance();cell,cfg=case(args.index)
    original=config(); restored=deepcopy(cfg);restored['training']['seed']=2026
    assert restored==original
    output=HERE/'runs'/f"{cell['run_id']}.json"; artifact=HERE/'artifacts'/cell['run_id']
    if output.exists() or artifact.exists(): raise FileExistsError(cell['run_id'])
    if args.preflight_only:
        print(json.dumps({'status':'preflight_passed','run_id':cell['run_id'],'seed':cfg['training']['seed'],'git_commit':commit,'only_scientific_change':'seed','test_evaluation_count':0}));return
    artifact.mkdir(parents=True)
    payload=dict(cell,experiment_name='target_multiseed_confirmation',scientific_run=True,status='running',git_commit=commit,git_branch='codex/close-experimental-stage',base_main_commit='93053d86eb3b7516ecf748fe060baa179d4e678b',dataset='KuaiRand-Pure',protocol='B',fingerprint=cfg['fingerprint'],evaluation_split='validation',test_usage='forbidden',test_evaluation_count=0,loss_formula=cfg['loss_formula'],loss_weight_mode=cfg['loss_weight_mode'],lambda_aux=cfg['optimization']['lambda_aux'],individual_aux_weights=weights(cell['active_targets']),auxiliary_loss_weights=weights(cell['active_targets']),effective_pos_weights={t:cfg['optimization']['effective_pos_weights'][t] for t in cell['active_targets']},seed=cfg['training']['seed'],train_batch_size=cfg['training']['train_batch_size'],eval_batch_size=cfg['training']['eval_batch_size'],optimization=cfg['optimization'],max_epochs=cfg['training']['max_epochs'],early_stopping_patience=cfg['training']['early_stopping_patience'],frozen_screening_code='599bcdb6e50dceea73233169b8834eaceef083a8',config_sha256=sha(ROOT/'experiments/target_combination_analysis/config.yaml'),source_manifest_sha256=sha(HERE/'frozen_source_hashes.json'),effective_config=cfg,start_time=frozen.now(),hostname=socket.gethostname(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),history=[])
    guard=DataAccessGuard(['/home/daryumin/iberdov/diplom/data',cfg['source']['validation_only_recbole_dir']]);guard.install()
    started=time.monotonic()
    try:
        frozen.execute(payload,cfg,artifact,False,guard)
        payload['status']='completed'
    except BaseException as exc:
        import traceback
        payload['status']='failed';payload['failure']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        raise
    finally:
        payload.update(end_time=frozen.now(),runtime_sec=time.monotonic()-started,data_access=guard.record())
        write(output,payload)

if __name__=='__main__': main()
