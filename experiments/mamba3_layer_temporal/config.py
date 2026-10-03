"""Frozen MIMO dual configuration; only layer sharing and output paths differ."""
import copy
from pathlib import Path
from experiments.mamba3_mimo_time.records import read
from experiments.mamba3_mimo_time.confirmation.config import CORE, PIN, POLICY, POLICY_SHA, ADMISSION, ADMISSION_SHA, SMOKE as OLD_SMOKE, SMOKE_SHA, PILOT_COMMIT, PILOT_SOURCE

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BRANCH='exp/mamba3-layer-temporal-functions'
PUBLICATION='0afbba24c9402886483d7f3cc2680222264052d9'
STUDY='mamba3_layer_temporal_pilot_001'
EXECUTION_ATTEMPT='001'
MANIFEST=HERE/'source_manifest.json'
PARENT_MANIFEST=ROOT/'experiments/mamba3_gap_trap/centered/confirmation/source_manifest.json'
PILOT=ROOT/'experiments/mamba3_mimo_time/runs/attempt_003/mamba3_mimo_dual_seed2026_001.json'
MODES=('shared_layers','layer_specific')
COUNTS=dict(shared_layers=715020,layer_specific=715152)
LOGS,RUNS=HERE/'slurm_logs/attempt_001',HERE/'runs/attempt_001'
LOGIN,RESERVATION,SUBMISSION=(LOGS/n for n in ('login_verification.json','reservation.json','submission.json'))
PIPELINE=LOGS/'pipeline_status.json'
INHERITED,GATE,SMOKE,SUMMARY=(RUNS/n for n in ('inherited_kernel.json','targeted_gate.json','smoke.json','pilot_summary.json'))
LAUNCHER=ROOT/'slurm/mamba3_layer_temporal.sh'


def paths(variant):
    if variant not in MODES:raise ValueError('Unplanned layer sharing variant')
    name=f'mamba3_layer_temporal_{variant}_seed2026_001';runtime=LOGS/name
    return dict(run_id=name,runtime=runtime,result=RUNS/(name+'.json'),lock=runtime/'run.lock',
                checkpoint=runtime/'checkpoints/best_state_dict.pth',metadata=runtime/'checkpoints/best_metadata.json')


def settings(variant,device=None,checkpoint_dir=None):
    p=paths(variant);values=copy.deepcopy(read(PILOT)['config'])
    values.update(temporal_sharing=variant,checkpoint_dir=str(checkpoint_dir or p['checkpoint'].parent))
    if device is not None:
        if device not in ('cpu','cuda'):raise ValueError(device)
        values.update(use_gpu=device=='cuda',device=device)
        if device=='cpu':values['gpu_id']=''
    return values


def plan():
    value=read(HERE/'study_plan.json')
    expected=dict(study_id=STUDY,modes=list(MODES),parameter_counts=COUNTS,seed=2026,max_scientific_fits=2,
                  max_jobs=1,test_evaluations=0,automatic_retries=0,reference_ms=838393,mode='dual',layers=2,heads=2,
                  tasks=[dict(temporal_sharing=v,seed=2026,run_id=paths(v)['run_id']) for v in MODES])
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Study plan drift')
    return value


def parameter_keys(variant):
    keys=plan()['common_parameter_keys']
    return keys+(['layer1_times.'+k[len('times.'):] for k in keys if k.startswith('times.')] if variant=='layer_specific' else [])


def unused():
    candidates=[LOGIN,RESERVATION,SUBMISSION,PIPELINE,INHERITED,GATE,SMOKE,SUMMARY,SUMMARY.with_suffix('.md'),LOGS/'pipeline.lock',LOGS/'gate',LOGS/'smoke']
    for v in MODES:candidates.extend(p for k,p in paths(v).items() if k!='run_id')
    occupied=[str(p) for p in candidates if p.exists() or p.is_symlink()]
    if occupied:raise FileExistsError('Study already owned; no retry: '+repr(occupied))
