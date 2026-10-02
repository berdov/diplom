"""Frozen eight-fit plan and one physical path per logical scientific run."""
import copy
from pathlib import Path
from .. import config as pilot
from experiments.mamba3_mimo_time.records import read, sha

HERE = Path(__file__).resolve().parent
ROOT = pilot.ROOT
PUBLICATION = 'e8f8f1ecce0646c5815b524d0bd9683c423b4775'
BRANCH = 'exp/mamba3-gap-trap-centered-confirmation'
STUDY = 'mamba3_gap_trap_centered_confirmation_001'
SEEDS = (2027, 2028, 2029, 2030)
MODES, COUNTS = pilot.MODES, pilot.COUNTS
CORE, PIN, POLICY_SHA = pilot.CORE, pilot.PIN, pilot.POLICY_SHA
PILOT_COMMIT, PILOT_SOURCE = pilot.PILOT_COMMIT, pilot.PILOT_SOURCE
ADMISSION_SHA, SMOKE_SHA = pilot.ADMISSION_SHA, pilot.SMOKE_SHA
PILOT = pilot.paths('fixed_replay')['result']
PARENT_EXECUTION = 'e4e31370e29c2a34c5f0f1071046fccebd77c289'
PARENT_SOURCE = '638b836780af7b6f9065546c128cb8ac2451c5d8af3581855029eea515c2ca10'
PLAN, MANIFEST, INDEX = (HERE/n for n in ('study_plan.json','source_manifest.json','source_index.json'))
LAUNCHER = ROOT/'slurm/mamba3_gap_trap_centered_confirmation.sh'
RUNS, LOGS = HERE/'runs', HERE/'slurm_logs'


def pilot_path(variant):
    return pilot.paths(variant)['result']


def historical(seed):
    if seed not in SEEDS:raise ValueError('Unplanned historical seed')
    return ROOT/f'experiments/mamba3_mimo_time/confirmation/runs/mamba3_mimo_confirm_dual_seed{seed}_001.json'


def paths(variant, seed):
    if variant not in MODES or seed not in SEEDS:raise ValueError('Only the eight preregistered fits')
    label = 'fixed' if variant=='fixed_replay' else 'centered'
    name = f'mamba3_gaptrap_centered_confirm_{label}_seed{seed}_001'
    runtime = LOGS/'fits'/name
    return dict(run_id=name,runtime=runtime,result=RUNS/(name+'.json'),lock=runtime/'run.lock',
                checkpoint=runtime/'checkpoints/best_state_dict.pth',metadata=runtime/'checkpoints/best_metadata.json')


def tasks():
    return [dict(variant=v,seed=s,run_id=paths(v,s)['run_id']) for s in SEEDS for v in MODES]


def allocation(attempt):
    if attempt not in ('001','002'):raise ValueError('At most one continuation')
    logs=LOGS/('attempt_'+attempt)
    return dict(attempt=attempt,logs=logs,**{k:logs/n for k,n in dict(login='login_verification.json',
        reservation='reservation.json',submission='submission.json',pipeline='pipeline_status.json',
        lock='pipeline.lock',inherited='inherited_admission.json',summary='confirmation_summary.json',
        terminal='terminal_metadata.json').items()})


def settings(variant, seed, device=None, checkpoint_dir=None):
    p=paths(variant,seed)
    values=copy.deepcopy(read(pilot_path(variant))['config'])
    values.update(seed=seed,checkpoint_dir=str(checkpoint_dir or p['checkpoint'].parent))
    if device is not None:
        if device not in ('cpu','cuda'):raise ValueError(device)
        values.update(device=device,use_gpu=device=='cuda')
        if device=='cpu':values['gpu_id']=''
    return values


def expected_plan():
    old=pilot.plan()
    return dict(study_id=STUDY,seeds=list(SEEDS),modes=list(MODES),tasks=tasks(),parameter_counts=COUNTS,
        primary_contrast=['centered_gap_trap','fixed_replay'],max_scientific_fits=8,max_jobs=2,
        allocations={'001':28800,'002':28800},automatic_retries=0,test_evaluations=0,
        save_margin_seconds=600,min_remaining_to_start_fit_seconds=5400,min_scheduler_poll_seconds=600,
        resources=dict(partition='rocky',account='proj_1833',constraint='type_e',gpu='a100:1',cpus=4,mem=0,requeue=False),
        reference_ms=838393,centered_formula='q=(g-R0)/(g+R0); T_prime=T+alpha*q',alpha_bounds=[0.,1.],
        alpha='One fp32 scalar shared across heads/layers; exact0 init; unchanged Adam and projection',
        masks=dict(first_event=0,padding=0,active_zero_gap=-1,target_timestamp_used=False),
        diagnostic_grid=old['diagnostic_grid'],diagnostic_content_logits=old['diagnostic_content_logits'],
        pilot_execution=PARENT_EXECUTION,pilot_source_hash=PARENT_SOURCE,pilot_job='4371876',
        inherited_gate_sha256=sha(pilot.GATE),inherited_smoke_sha256=sha(pilot.SMOKE),
        historical_fixed={str(s):dict(path=str(historical(s).relative_to(ROOT)),sha256=sha(historical(s))) for s in SEEDS},
        config_sources={v:str(pilot_path(v).relative_to(ROOT)) for v in MODES},allowed_config_changes=['seed','checkpoint_dir'],
        dataset=dict(manifest_sha256=read(PILOT)['manifest_sha256'],train_time_stats_sha256=read(PILOT)['train_time_stats_sha256'],protocol=read(PILOT)['protocol']),
        selection=old['selection'],first27='Only full observed epochs0..26; paired available subset, descriptive',
        source_identity='source_manifest.json SHA256 digest of exact files, including this plan; no self-referential hash',
        subsets=dict(primary=list(SEEDS),secondary_including_exploratory_pilot=[2026,*SEEDS]),
        aggregation='sample std ddof=1, paired signs and deltas, relative difference of means; no p-values',
        replay='Exact historical fixed scientific history/loss/metrics/state/RNG/first batch/checkpoint; timing/memory excluded',
        continuation='Once only after terminal safe deadline guard between fits, no technical failure or started-incomplete fit; unchanged execution/source/config; never-started suffix only',
        no_gpu_gate_repeat=True,no_extra_forward=True,no_tuning=True,point4='NOT_STARTED',article='UNCHANGED')


def plan():
    value=read(PLAN)
    if value!=expected_plan():raise ValueError('Frozen confirmation plan drift')
    return value


def index():
    expected=dict(study_id=STUDY,entries=[dict(**t,result=str(paths(t['variant'],t['seed'])['result'].relative_to(ROOT))) for t in tasks()])
    if read(INDEX)!=expected:raise ValueError('Frozen source index drift')
    return expected


def unused(attempt, remaining):
    a=allocation(attempt)
    candidates=[a[k] for k in ('login','reservation','submission','pipeline','lock','inherited','summary','terminal')]
    for task in remaining:candidates.extend(paths(task['variant'],task['seed'])[k] for k in ('result','runtime','lock','checkpoint','metadata'))
    if any(p.exists() or p.is_symlink() for p in candidates):raise FileExistsError('Already owned; no overwrite or retry')
