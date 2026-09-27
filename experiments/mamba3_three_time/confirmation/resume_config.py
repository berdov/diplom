"""Attempt003 owns seven new executions; logical study and parent bytes stay fixed."""
import json
from .config import HERE, ROOT, SEEDS, MODES, task, plan as study_plan

PARENT_JOB = '4355052'
PARENT_COMMIT = '3d3b305c4c74e208cd226d3ed4ccec37b4fd8311'
PARENT_HASH = '0e93a9aec6d152f5f13dd5ccae30d8a761c09e552aecc4d7d884e573da8f7785'
ARCHIVE = HERE / 'evidence/job4355052'
PLAN = HERE / 'resume_plan_003.json'
LINEAGE = HERE / 'resume_lineage_003.json'
LOGS = HERE / 'slurm_logs/attempt_003'
RUNS = HERE / 'runs/attempt_003'
SUBMISSION = LOGS / 'submission_003.json'
LOGIN = LOGS / 'login_verification.json'
LOCK = LOGS / 'pipeline.lock'
SUMMARY = RUNS / 'confirmation_summary.json'
ORDER = (('triple', 2027), *((m, s) for s in (2028, 2029, 2030) for m in MODES))


def execution_task(mode, seed):
    value = task(mode, seed)
    if (mode, seed) not in ORDER:
        raise ValueError('Not one of seven authorized remaining fits')
    return value


def paths(mode, seed):
    t = execution_task(mode, seed)
    runtime = LOGS / t['run_id']
    return dict(runtime=runtime, result=RUNS / (t['run_id'] + '.json'), lock=runtime / 'run.lock',
                checkpoint=runtime / 'checkpoints/best_state_dict.pth', metadata=runtime / 'checkpoints/best_metadata.json')


def plan():
    study_plan()
    row = json.loads(PLAN.read_text())
    if (row['parent_job'], row['parent_execution_commit'], row['parent_source_hash']) != (PARENT_JOB, PARENT_COMMIT, PARENT_HASH):
        raise ValueError('Unexpected resume parent')
    if row['tasks'] != [execution_task(m, s) for m, s in ORDER] or row['max_scientific_fits'] != 7:
        raise ValueError('Seven-fit resume order changed')
    index = row['source_index']
    if [(r['seed'], r['mode']) for r in index] != [(s, m) for s in (2026, *SEEDS) for m in MODES]:
        raise ValueError('Exactly ten predetermined result sources required')
    for r in index:
        s, m = r['seed'], r['mode']
        expected = (HERE.parent / f'validation_pilot/runs/mamba3_three_time_siso_{m}_seed2026_001.json' if s == 2026
                    else HERE / f'runs/{task(m, s)["run_id"]}.json' if (m, s) == ('dual', 2027)
                    else paths(m, s)['result'])
        kind = 'PILOT' if s == 2026 else 'REUSED_COMPLETED' if (m, s) == ('dual', 2027) else 'NEW'
        if r['path'] != str(expected.relative_to(ROOT)) or r['kind'] != kind:
            raise ValueError('Source index path/kind changed')
    return row


def unused(include_submission=False):
    targets = [LOCK, LOGS/'pipeline_status.json', SUMMARY, SUMMARY.with_suffix('.md'), SUMMARY.with_suffix('.svg')]
    for m, s in ORDER:
        targets.extend(paths(m, s)[k] for k in ('result', 'lock', 'checkpoint', 'metadata'))
    if include_submission:
        targets.extend((SUBMISSION, LOGIN))
    occupied = [str(p) for p in targets if p.exists() or p.is_symlink()]
    if occupied:
        raise FileExistsError('Resume already reserved/executed; no retry: ' + str(occupied))
