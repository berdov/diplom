"""Authorization for this one explicit infrastructure retry, not a retry engine."""
from . import config as c
from .records import read, sha, digest, accepted_cases

POLICY_SHA = '216fbde21dcd5d349857bef8ef963e535d92aefee1e971e0de91d67385de43b7'
PLAN_SHA = '15240112901bc84f643faae4059fb5f1a6a93418f843d1bd94f5dde3e32ddc4f'
ANCESTOR_SHA = '5c51d7628c109f8011d58bc3a629d3e52a1ca0aa78f2caecda1f6b242356e672'


def parent_binding():
    return dict(execution_attempt=c.EXECUTION_ATTEMPT, retry_of_job=c.PARENT_JOB,
                parent_execution_commit=c.PARENT_COMMIT, parent_source_hash=c.PARENT_SOURCE,
                parent_failure_evidence_sha256=c.PARENT_EVIDENCE_SHA,
                reason='gradient_capture_hook_active_during_no_grad_intervention', parent_scientific_fits=0)


def verify_parent(root=None, live=False):
    root = c.ROOT if root is None else root
    here = root / c.HERE.relative_to(c.ROOT)
    # Only these two terminated and preserved attempts authorize attempt 003.
    ancestors = [('4356310', ANCESTOR_SHA, 'f4938792ab3415e431320d5ca7c2d6ce042ab35a',
                  '52c903ac4d2212ae8d81645079e424265b30c0f0733a3b48ec70571de8f9fc12'),
                 (c.PARENT_JOB, c.PARENT_EVIDENCE_SHA, c.PARENT_COMMIT, c.PARENT_SOURCE)]
    if sha(root/c.POLICY.relative_to(c.ROOT)) != POLICY_SHA or sha(here/'study_plan.json') != PLAN_SHA:
        raise ValueError('Scientific plan/policy changed')
    for job, manifest_sha, commit, source in ancestors:
        _verify_failure(root, here/'evidence'/('job'+job), job, manifest_sha, commit, source, live)
    if live:
        for base in (here/'slurm_logs', here/'runs'):
            for directory in base.glob('attempt_*'):
                if directory.name not in ('attempt_002', 'attempt_' + c.EXECUTION_ATTEMPT):
                    raise ValueError('Unknown attempt blocks retry: ' + str(directory))
    return parent_binding()


def _verify_failure(root, evidence, job, manifest_sha, commit, source, live):
    manifest_path = evidence/'preservation_manifest.json'
    if sha(manifest_path) != manifest_sha:
        raise ValueError('Parent preservation manifest changed')
    manifest = read(manifest_path)
    if (manifest['job_id'] != job or manifest['execution_commit'] != commit
            or manifest['source_hash'] != source or manifest['scientific_fits'] != 0
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
    if digest(old_manifest['files']) != source or old_manifest['source_hash'] != source:
        raise ValueError('Parent source identity changed')
    pipeline = read(evidence/'slurm_logs/pipeline_status.json')
    summary = read(evidence/'runs/pilot_summary.json')
    submitted = read(evidence/'slurm_logs/submission_001.json')
    stages = [('preflight', 'FAIL')] if job == '4356310' else [('preflight', 'PASS'), ('admission', 'FAIL')]
    if (pipeline['job_id'] != job or submitted['job_id'] != job
            or pipeline['execution_commit'] != commit or pipeline['source_hash'] != source
            or submitted['jobs_submitted'] != 1 or pipeline['status'] != 'FAIL'
            or [(s['stage'], s['status']) for s in pipeline['stages']] != stages
            or pipeline['scientific_fits'] != 0 or pipeline['scientific_fits_completed'] != 0
            or summary['scientific_fits_started'] != 0 or summary['scientific_fits_completed'] != 0
            or summary['test_evaluation_count'] != 0 or pipeline['test_evaluation_count'] != 0):
        raise ValueError('Parent progressed beyond the authorized infrastructure failure')
    if any(r['status'] != 'NOT_RUN' or r['actual_epochs'] != 0 or r['metrics'] is not None for r in summary['rows']):
        raise ValueError('Parent failure/summary mismatch')
    if job == '4356310':
        if 'CPU preflight initialized CUDA' not in (evidence/'slurm_logs/preflight/stderr.log').read_text():
            raise ValueError('Ancestor CPU failure mismatch')
    else:
        admission = read(evidence/'runs/admission_001.json')
        preflight = read(evidence/'slurm_logs/preflight/evidence.json')
        specs = read(evidence/'study_plan.json')['required_cases']
        cases = admission['cases']
        if (admission['job_id'] != job or admission['execution_commit'] != commit
                or admission['source_hash'] != source or admission['status'] != 'FAIL'
                or admission['training_authorized'] is not False or admission['scientific_fits'] != 0
                or admission['test_evaluation_count'] != 0 or len(cases) != 29
                or not accepted_cases(cases[:28], specs[:28])
                or cases[-1]['case_id'] != 'prefix_base_L17_P7_x1' or cases[-1]['status'] != 'FAIL'
                or cases[-1]['checks'] or "can't retain_grad on Tensor that has requires_grad=False" not in cases[-1].get('traceback', '')
                or preflight['status'] != 'PASS' or preflight['final_cuda_initialized'] is not False
                or manifest['ancestor_evidence_sha256'] != ANCESTOR_SHA):
            raise ValueError('Parent hook failure mismatch')
    if live:
        for name in manifest['absent_paths']:
            p = root/name
            if p.exists() or p.is_symlink():
                raise ValueError('Parent scientific/gate artifact appeared: ' + name)
        old_logs = root/(c.HERE/'slurm_logs').relative_to(c.ROOT)
        if job != '4356310':
            old_logs = old_logs/'attempt_002'
        if any((old_logs/c.paths(m)['run_id']).exists() for m in c.MODES):
            raise ValueError('Unexpected parent scientific runtime directory')
