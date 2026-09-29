"""Only seed and output paths differ from the successful per-mode pilot config."""
import copy
from pathlib import Path
from .records import read

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PILOT_ROOT = HERE.parent
PILOT_COMMIT = 'c1dd31eec7907c67348769b6a1811b89aa4011c0'
PILOT_SOURCE = '3182ed3ede715c01190e14d63c3b7c2613fc647d7a17cd871655fb16189c7940'
PUBLICATION = '9a01d220efa64b9d0fe6b4a7e69b948ebbe16f07'
BRANCH = 'exp/mamba3-mimo-confirmation'
STUDY = 'mamba3_mimo_confirmation_001'
MODES, SEEDS = ('base', 'dual', 'triple'), (2027, 2028, 2029, 2030)
COUNTS = dict(base=714888, dual=715020, triple=715086)
CORE = '460bd3e687d26a458cafc21960391f83905da962ca5c992d6062c7b048f0a73f'
PIN = 'e9594ce1c732d97440f0332fdc43170a2294dbfa'
POLICY = PILOT_ROOT / 'mimo_numeric_acceptance_v1.json'
POLICY_SHA = '216fbde21dcd5d349857bef8ef963e535d92aefee1e971e0de91d67385de43b7'
ADMISSION = PILOT_ROOT / 'runs/attempt_003/admission_001.json'
SMOKE = PILOT_ROOT / 'runs/attempt_003/smoke_001.json'
ADMISSION_SHA = 'c673524fda1a41d472c6cafebca7e25dad5cd7bd5df99fa772dfdcd4ee8bd5bc'
SMOKE_SHA = 'ac17c67b3fc411fce5cbdd53a855e75392d5bfd953b29a434e7237f135e7db9b'
LOGS, RUNS = HERE / 'slurm_logs', HERE / 'runs'
LOGIN, RESERVATION = LOGS / 'login_verification_001.json', LOGS / 'reservation_001.json'
SUBMISSION, PIPELINE = LOGS / 'submission_001.json', LOGS / 'pipeline_status.json'
INHERITED, SUMMARY = RUNS / 'inherited_admission_001.json', RUNS / 'confirmation_summary.json'
LAUNCHER = ROOT / 'slurm/mamba3_mimo_confirmation.sh'


def pilot_path(mode):
    if mode not in MODES:
        raise ValueError(mode)
    return PILOT_ROOT / f'runs/attempt_003/mamba3_mimo_{mode}_seed2026_001.json'


def paths(mode, seed):
    if mode not in MODES or seed not in SEEDS:
        raise ValueError('Only the twelve preregistered fresh fits')
    name = f'mamba3_mimo_confirm_{mode}_seed{seed}_001'
    runtime = LOGS / name
    return dict(run_id=name, runtime=runtime, result=RUNS / (name + '.json'), lock=runtime / 'run.lock',
                checkpoint=runtime / 'checkpoints/best_state_dict.pth', metadata=runtime / 'checkpoints/best_metadata.json')


def tasks():
    return [dict(mode=m, seed=s, run_id=paths(m, s)['run_id']) for s in SEEDS for m in MODES]


def settings(mode, seed, device=None, checkpoint_dir=None):
    if seed not in (2026, *SEEDS):
        raise ValueError('Unplanned seed')
    if seed == 2026 and checkpoint_dir is None:
        raise ValueError('Seed2026 is comparison-only; explicit temporary output path required')
    values = copy.deepcopy(read(pilot_path(mode))['config'])
    values.update(seed=seed, checkpoint_dir=str(checkpoint_dir if checkpoint_dir is not None else paths(mode, seed)['checkpoint'].parent))
    if device is not None:
        if device not in ('cpu', 'cuda'):
            raise ValueError(device)
        values.update(use_gpu=device == 'cuda', device=device)
        if device == 'cpu':
            values['gpu_id'] = ''
    return values


def expected_plan():
    return dict(study_id=STUDY, tasks=tasks(), parameter_counts=COUNTS,
                primary='triple - dual', contrasts=[['triple','dual'], ['dual','base'], ['triple','base']],
                budget=dict(allocation_seconds=28800, save_margin_seconds=600, min_remaining_to_start_fit_seconds=5400),
                resources=dict(partition='rocky', account='proj_1833', constraint='type_e', gpu='a100:1', cpus=4, mem=0, time='08:00:00', requeue=False),
                max_scientific_fits=12, max_jobs=1, test_evaluations=0, automatic_retries=False,
                setup_order=['runtime/config','seed','dataset/loaders','model','trainer','fit'],
                selection='Last tied maximum rounded VALID NDCG@10; unchanged RecBole.fit',
                config_sources={m:str(pilot_path(m).relative_to(ROOT)) for m in MODES},
                inherited_admission_sha256=ADMISSION_SHA, inherited_smoke_sha256=SMOKE_SHA,
                policy_sha256=POLICY_SHA, pilot_execution=PILOT_COMMIT, pilot_job='4358147',
                subsets=dict(confirmatory=list(SEEDS), with_exploratory_pilot=[2026,*SEEDS]),
                first27='Observed history only; complete iff actual_epochs >= 27',
                stop_rule='Technical failure or deadline only; never low VALID metric',
                fresh_process=True, fresh_weights=True, allowed_config_changes=['seed','checkpoint_dir'],
                uncertainty='sample std ddof=1; no automatic significance or backbone selection')


def plan():
    value = read(HERE / 'study_plan.json')
    if value != expected_plan():
        raise ValueError('Frozen confirmation plan drift')
    return value


def unused():
    candidates = [LOGIN, RESERVATION, SUBMISSION, INHERITED, SUMMARY, SUMMARY.with_suffix('.md'), PIPELINE, LOGS/'pipeline.lock']
    for task in tasks():
        candidates += [v for k,v in paths(task['mode'],task['seed']).items() if k in ('result','lock','checkpoint','metadata')]
    occupied = [str(p) for p in candidates if p.exists() or p.is_symlink()]
    if occupied:
        raise FileExistsError('Already reserved/executed; no automatic retry: ' + repr(occupied))
