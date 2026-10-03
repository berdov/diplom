"""Historical dual configuration; only memory readout and output paths differ."""
import copy
import os
from pathlib import Path
from experiments.mamba3_mimo_time.records import read
from experiments.mamba3_mimo_time.confirmation.config import CORE,PIN,POLICY,POLICY_SHA,ADMISSION,ADMISSION_SHA,SMOKE as OLD_SMOKE,SMOKE_SHA,PILOT_COMMIT,PILOT_SOURCE

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BRANCH='exp/mamba3-time-addressed-memory'
PUBLICATION='7ba5e45ed110914b354c89d4377836910b55408c'
STUDY='mamba3_time_addressed_memory_pilot_001'
EXECUTION_ATTEMPT=os.environ.get('TIME_MEMORY_ATTEMPT','001')
if EXECUTION_ATTEMPT not in ('001','002'):raise ValueError('At most initial allocation and one pre-fit technical retry')
MANIFEST=HERE/('source_manifest.json' if EXECUTION_ATTEMPT=='001' else 'source_manifest_002.json')
PARENT_MANIFEST=ROOT/'experiments/mamba3_layer_temporal/source_manifest.json'
PILOT=ROOT/'experiments/mamba3_mimo_time/runs/attempt_003/mamba3_mimo_dual_seed2026_001.json'
MODES=('no_memory','index_memory','time_memory')
COUNTS=dict(no_memory=715020,index_memory=715021,time_memory=715021)
LOGS,RUNS=HERE/('slurm_logs/attempt_'+EXECUTION_ATTEMPT),HERE/('runs/attempt_'+EXECUTION_ATTEMPT)
LOGIN,RESERVATION,SUBMISSION,PIPELINE=(LOGS/n for n in ('login_verification.json','reservation.json','submission.json','pipeline_status.json'))
COVERAGE=LOGS/'train_coverage.json'
INHERITED,GATE,SMOKE,SUMMARY=(RUNS/n for n in ('inherited_kernel.json','targeted_gate.json','smoke.json','pilot_summary.json'))
LAUNCHER=ROOT/'slurm/mamba3_time_memory.sh'


def paths(variant):
    if variant not in MODES:raise ValueError('Unplanned memory mode')
    name=f'mamba3_time_memory_{variant}_seed2026_001';runtime=LOGS/name
    return dict(run_id=name,runtime=runtime,result=RUNS/(name+'.json'),lock=runtime/'run.lock',checkpoint=runtime/'checkpoints/best_state_dict.pth',metadata=runtime/'checkpoints/best_metadata.json')


def settings(variant,device=None,checkpoint_dir=None):
    p=paths(variant);values=copy.deepcopy(read(PILOT)['config'])
    values.update(memory_mode=variant,checkpoint_dir=str(checkpoint_dir or p['checkpoint'].parent))
    if device is not None:
        if device not in ('cpu','cuda'):raise ValueError(device)
        values.update(use_gpu=device=='cuda',device=device)
        if device=='cpu':values['gpu_id']=''
    return values


def plan():
    value=read(HERE/'study_plan.json')
    expected=dict(study_id=STUDY,modes=list(MODES),parameter_counts=COUNTS,seed=2026,max_scientific_fits=3,max_jobs=2,
        test_evaluations=0,automatic_continuations=0,reference_ms=838393,mode='dual',layers=2,heads=2,history=50,K=4,anchors=[1,4,16,32],
        tasks=[dict(memory_mode=v,seed=2026,run_id=paths(v)['run_id']) for v in MODES])
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Study plan drift')
    return value


def parameter_keys(variant):
    return plan()['common_parameter_keys']+(['beta'] if variant!='no_memory' else [])


def unused():
    candidates=[LOGIN,RESERVATION,SUBMISSION,PIPELINE,INHERITED,GATE,SMOKE,SUMMARY,LOGS/'pipeline.lock',LOGS/'gate',LOGS/'smoke']
    for v in MODES:candidates.extend(p for k,p in paths(v).items() if k!='run_id')
    occupied=[str(p) for p in candidates if p.exists() or p.is_symlink()]
    if occupied:raise FileExistsError('Study paths already reserved; no duplicate: '+repr(occupied))
