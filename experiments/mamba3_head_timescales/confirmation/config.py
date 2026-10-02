"""Frozen twelve-run scope; explicit physical allocation paths."""
import copy
from pathlib import Path
from experiments.mamba3_mimo_time.records import read
from experiments.mamba3_head_timescales.config import CORE, PIN, POLICY, POLICY_SHA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PILOT_ROOT = HERE.parent
PUBLICATION = 'a45e181471c511cc600ed829f6f6a70957a330d8'
PILOT_COMMIT = '4724392c88a2298e57fa662cba33baa3ab9ecdbb'
PILOT_SOURCE = '0735c8689e3a0f962a0be7560c08d8ccbc6cf53089a6457151b8e8af40fd899e'
BRANCH = 'exp/mamba3-head-timescales-confirmation'
STUDY = 'mamba3_head_timescales_confirmation_001'
MODES = ('fixed', 'shared_tau', 'head_tau')
SEEDS = (2027, 2028, 2029, 2030)
COUNTS = dict(fixed=715020, shared_tau=715022, head_tau=715024)
GATE = PILOT_ROOT/'runs/attempt_002/targeted_gate_001.json'
SMOKE = PILOT_ROOT/'runs/attempt_002/smoke_001.json'
GATE_SHA = '22a221b56a9cec4a88493c2827a943af65a46592f69a3deaabf14bc72992f51f'
SMOKE_SHA = '32b631ba67656ea4312388e9d35d662615d9b8820c87be10aba627e9912e3e93'
LAUNCHER = ROOT/'slurm/mamba3_head_timescales_confirmation.sh'
RUNTIME = HERE/'runtime'


def pilot_path(variant):
    if variant not in MODES:
        raise ValueError('Unplanned variant')
    return PILOT_ROOT/f'runs/attempt_002/mamba3_headtime_{variant}_seed2026_001.json'


def allocation(attempt='001'):
    if attempt not in ('001', '002'):
        raise ValueError('At most two allocations')
    logs, runs = HERE/f'slurm_logs/attempt_{attempt}', HERE/f'runs/attempt_{attempt}'
    result = dict(attempt=attempt, logs=logs, runs=runs,
                  manifest=HERE/('source_manifest.json' if attempt=='001' else 'source_manifest_002.json'),
                  source_index=HERE/('source_index.json' if attempt=='001' else 'source_index_002.json'))
    result.update({key:logs/name for key,name in dict(login='login_verification.json',reservation='reservation.json',
                  submission='submission.json',pipeline='pipeline_status.json',lock='pipeline.lock').items()})
    result.update(inherited=runs/'inherited_admission.json',summary=runs/'confirmation_summary.json')
    return result


def paths(variant, seed, attempt='001'):
    if variant not in MODES or seed not in SEEDS:
        raise ValueError('Only twelve preregistered fresh fits; no seed2026 run')
    a = allocation(attempt)
    name = f'mamba3_headtime_confirm_{variant}_seed{seed}_001'
    runtime = a['logs']/name
    return dict(run_id=name,runtime=runtime,result=a['runs']/(name+'.json'),lock=runtime/'run.lock',
                checkpoint=runtime/'checkpoints/best_state_dict.pth',metadata=runtime/'checkpoints/best_metadata.json')


def tasks():
    return [dict(variant=v,seed=s,run_id=paths(v,s)['run_id']) for s in SEEDS for v in MODES]


def settings(variant,seed,attempt='001',device=None,checkpoint_dir=None):
    if seed not in (2026,*SEEDS) or (seed==2026 and (checkpoint_dir is None or device!='cpu')):
        raise ValueError('Seed2026 only CPU setup replay with explicit temporary outputs')
    values = copy.deepcopy(read(pilot_path(variant))['config'])
    directory = checkpoint_dir if checkpoint_dir is not None else paths(variant,seed,attempt)['checkpoint'].parent
    values.update(seed=seed,checkpoint_dir=str(directory))
    if device is not None:
        if device not in ('cpu','cuda'):
            raise ValueError(device)
        values.update(use_gpu=device=='cuda',device=device)
        if device=='cpu':values['gpu_id']=''
    return values


def expected_plan():
    old = read(PILOT_ROOT/'study_plan.json')
    return dict(study_id=STUDY,tasks=tasks(),modes=list(MODES),seeds=list(SEEDS),parameter_counts=COUNTS,
        primary=['head_tau','shared_tau'],contrasts=[['head_tau','shared_tau'],['head_tau','fixed'],['shared_tau','fixed']],
        subsets=dict(confirmatory=list(SEEDS),with_exploratory_pilot=[2026,*SEEDS]),
        max_scientific_fits=12,max_jobs=2,max_requested_gpu_seconds=43200,test_evaluations=0,
        allocations={'001':28800,'002':14400},save_margin_seconds=600,min_remaining_to_start_fit_seconds=5400,
        max_session_wait_seconds=57600,max_queue_wait_seconds=14400,min_scheduler_poll_seconds=600,
        resources=dict(partition='rocky',account='proj_1833',constraint='type_e',gpu='a100:1',cpus=4,mem=0,requeue=False),
        setup_order=['runtime/config','seed','dataset/loaders','model','trainer','fit'],
        config_sources={v:str(pilot_path(v).relative_to(ROOT)) for v in MODES},
        allowed_config_changes=['seed','checkpoint_dir'],diagnostic_grid=old['diagnostic_grid'],
        inherited_gate_sha256=GATE_SHA,inherited_smoke_sha256=SMOKE_SHA,policy_sha256=POLICY_SHA,
        pilot_execution=PILOT_COMMIT,pilot_source_hash=PILOT_SOURCE,pilot_job='4362620',
        mathematical_dependencies='Unchanged model/calibrators/trainer/kernels/data/numerical policy',
        selection='Last tied maximum rounded VALID NDCG@10; unchanged RecBole.fit and stopping_step10',
        first27='Only complete observed epochs0..26; each contrast has its own complete-pair subset',
        aggregation='sample std ddof1; paired signs/differences; relative gain on identical seed intersection; no p-values',
        continuation='Only once after terminal preserved allocation: never-started infrastructure/serialization or deadline-before-start; any started incomplete/numerical failure blocks continuation',
        no_metric_retries=True,fresh_process=True,fresh_weights=True,
        inherited_gpu_checks_repeated=0,extra_diagnostic_evaluations=0,
        limitations='Exploratory pilot separate; no automatic backbone selection or head specialization claim')


def plan():
    p=read(HERE/'study_plan.json')
    if p!=expected_plan():raise ValueError('Frozen plan drift')
    return p


def index(attempt='001'):
    value=read(allocation(attempt)['source_index'])
    entries=value['entries']
    if value.get('study_id')!=STUDY or value.get('allocation_attempt')!=attempt or len(entries)!=12:
        raise ValueError('Source-index scope')
    for entry,task in zip(entries,tasks(),strict=True):
        if any(entry.get(k)!=v for k,v in task.items()):raise ValueError('Source-index task order/duplicates')
        physical=entry.get('attempt')
        if physical not in (('001',) if attempt=='001' else ('001','002')):raise ValueError('Source-index allocation')
        expected=str(paths(task['variant'],task['seed'],physical)['result'].relative_to(ROOT))
        if entry.get('result')!=expected:raise ValueError('Source-index path')
        if physical!=attempt and not entry.get('preserved'):raise ValueError('Missing previous allocation lineage')
        if physical==attempt and 'preserved' in entry:raise ValueError('Current source cannot bypass allocation ownership')
        if physical!=attempt and set(entry['preserved'])!={'result_sha256','metadata_sha256','checkpoint_sha256','job_id','execution_commit','source_hash','reservation_sha256'}:
            raise ValueError('Incomplete preserved source identity')
    if len({e['result'] for e in entries})!=12:raise ValueError('Duplicate physical result')
    return value


def unused(attempt='001'):
    a=allocation(attempt)
    candidates=[a[k] for k in ('login','reservation','submission','pipeline','lock','inherited','summary')]
    candidates += [a['summary'].with_suffix('.md')]
    for entry in index(attempt)['entries']:
        if entry['attempt']==attempt:
            p=paths(entry['variant'],entry['seed'],attempt)
            candidates += [p[k] for k in ('result','runtime','lock','checkpoint','metadata')]
    occupied=[str(p) for p in candidates if p.exists() or p.is_symlink()]
    if occupied:raise FileExistsError('Allocation already owned; no overwrite: '+repr(occupied))
