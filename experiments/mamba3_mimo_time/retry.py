"""Authorization for this one explicit infrastructure retry, not a retry engine."""
from . import config as c
from .records import read, sha, digest

POLICY_SHA = '216fbde21dcd5d349857bef8ef963e535d92aefee1e971e0de91d67385de43b7'
PLAN_SHA = '15240112901bc84f643faae4059fb5f1a6a93418f843d1bd94f5dde3e32ddc4f'


def parent_binding():
    return dict(execution_attempt=c.EXECUTION_ATTEMPT, retry_of_job=c.PARENT_JOB,
                parent_execution_commit=c.PARENT_COMMIT, parent_source_hash=c.PARENT_SOURCE,
                parent_failure_evidence_sha256=c.PARENT_EVIDENCE_SHA,
                reason='cpu_preflight_not_isolated_from_gpu', parent_scientific_fits=0)


def verify_parent(root=None, live=False):
    root = c.ROOT if root is None else root
    evidence = root / c.PARENT_EVIDENCE.relative_to(c.ROOT)
    manifest_path = evidence/'preservation_manifest.json'
    if sha(manifest_path) != c.PARENT_EVIDENCE_SHA:
        raise ValueError('Parent preservation manifest changed')
    manifest = read(manifest_path)
    if (manifest['job_id'] != c.PARENT_JOB or manifest['execution_commit'] != c.PARENT_COMMIT
            or manifest['source_hash'] != c.PARENT_SOURCE or manifest['scientific_fits'] != 0
            or manifest['state'] != 'FAILED' or manifest['exit_code'] != '1:0'):
        raise ValueError('Wrong parent failure')
    for row in manifest['files']:
        archive = root/row['destination']
        if not archive.resolve().is_relative_to(evidence.resolve()) or sha(archive) != row['sha256']:
            raise ValueError('Preserved parent evidence changed: ' + row['destination'])
        # Current infrastructure sources may change; original run artifacts may not.
        if live and ('/slurm_logs/' in row['source'] or '/runs/' in row['source']):
            if sha(root/row['source']) != row['sha256']:
                raise ValueError('Original parent run artifact changed: ' + row['source'])
    generated = manifest['generated_settings']
    if sha(root/generated['path']) != generated['sha256']:
        raise ValueError('Parent settings snapshot changed')
    old_manifest = read(evidence/'source_manifest.json')
    if digest(old_manifest['files']) != c.PARENT_SOURCE or old_manifest['source_hash'] != c.PARENT_SOURCE:
        raise ValueError('Parent source identity changed')
    if sha(root/c.POLICY.relative_to(c.ROOT)) != POLICY_SHA or sha(root/(c.HERE/'study_plan.json').relative_to(c.ROOT)) != PLAN_SHA:
        raise ValueError('Scientific plan/policy changed')
    pipeline = read(evidence/'slurm_logs/pipeline_status.json')
    summary = read(evidence/'runs/pilot_summary.json')
    submitted = read(evidence/'slurm_logs/submission_001.json')
    if (pipeline['job_id'] != c.PARENT_JOB or submitted['job_id'] != c.PARENT_JOB
            or submitted['jobs_submitted'] != 1 or pipeline['status'] != 'FAIL'
            or [(s['stage'], s['status']) for s in pipeline['stages']] != [('preflight', 'FAIL')]
            or pipeline['scientific_fits'] != 0 or pipeline['scientific_fits_completed'] != 0
            or summary['scientific_fits_started'] != 0 or summary['scientific_fits_completed'] != 0
            or summary['test_evaluation_count'] != 0 or pipeline['test_evaluation_count'] != 0):
        raise ValueError('Parent progressed beyond the authorized infrastructure failure')
    if ('CPU preflight initialized CUDA' not in (evidence/'slurm_logs/preflight/stderr.log').read_text()
            or any(r['status'] != 'NOT_RUN' or r['actual_epochs'] != 0 or r['metrics'] is not None for r in summary['rows'])):
        raise ValueError('Parent failure/summary mismatch')
    if live:
        old_logs = root/(c.HERE/'slurm_logs').relative_to(c.ROOT)
        for directory in old_logs.glob('attempt_*'):
            if directory.name != 'attempt_' + c.EXECUTION_ATTEMPT:
                if (directory/'pipeline.lock').exists() or list(directory.glob('submission_*.json')):
                    raise ValueError('Another attempt already owns this logical study')
        for name in manifest['absent_paths']:
            p = root/name
            if p.exists() or p.is_symlink():
                raise ValueError('Parent scientific/gate artifact appeared: ' + name)
        if any((root/(c.HERE/'slurm_logs').relative_to(c.ROOT)/c.paths(m)['run_id']).exists() for m in c.MODES):
            raise ValueError('Unexpected parent scientific runtime directory')
    return parent_binding()
