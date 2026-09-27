"""One durable resume reservation and one sbatch; no polling or automatic retry."""
import argparse
import json
import os
import re
import subprocess
import traceback
from . import config as c, provenance as p, resume_config as r, resume_provenance as q
from .submit import durable_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit',required=True)
    commit = parser.parse_args().commit
    if str(c.ROOT) != '/home/daryumin/iberdov/diplom':
        raise ValueError('Canonical cluster checkout required')
    r.unused(include_submission=True)
    login = q.login_verify(commit)
    m = p.verify()
    for name in ('login_preflight.json','cpu_tests.json','no_git_preflight.json'):
        value = q.read(r.LOGS/name)
        if value['status'] != 'PASS' or value['source_hash'] != m['source_hash']:
            raise ValueError('Same-source CPU/effective-config/no-Git preflight required')
    parent = q.parent_ready()
    durable_record(r.LOGIN,login)
    row = dict(**q.bindings(commit,m),status='RESERVED',reserved_at=p.now(),login_verification_sha256=p.sha(r.LOGIN),
               resume_reason='optimizer_metadata_JSON_tuple_list_mismatch_before_triple2027_fit',
               max_scientific_fits=7,previous_completed_confirmation_fits=1,new_diagnostic_optimizer_steps=0,
               new_diagnostic_batches=0,scientific_fits_before_submit=0,TEST='NOT_RUN',test_evaluation_count=0,
               inherited_gates=parent['inherited_gates'],jobs_submitted=None)
    durable_record(r.SUBMISSION,row)
    try:
        response = subprocess.run(['sbatch','--parsable','slurm/mamba3_three_time_confirmation_resume.sh'],cwd=c.ROOT,
            env=dict(os.environ,RUN_COMMIT=commit,EXPECTED_STUDY_HASH=m['source_hash'],EXPECTED_CORE_HASH=p.CORE),
            text=True,capture_output=True,check=False)
        row.update(stdout=response.stdout,stderr=response.stderr,returncode=response.returncode)
        match = re.fullmatch(r'([0-9]+)(?:;[A-Za-z0-9_.-]+)?',response.stdout.strip())
        if response.returncode != 0 or match is None:
            row['status'] = 'SUBMISSION_UNKNOWN_NO_RETRY'
            p.atomic_json(r.SUBMISSION,row)
            raise RuntimeError('Ambiguous/failed sbatch; do not repeat')
        row.update(status='SUBMITTED',job_id=match.group(1),jobs_submitted=1)
        p.atomic_json(r.SUBMISSION,row)
        print(json.dumps(row,indent=2),flush=True)
    except BaseException:
        if row['status']=='RESERVED':
            row.update(status='SUBMISSION_UNKNOWN_NO_RETRY',traceback=traceback.format_exc())
            p.atomic_json(r.SUBMISSION,row)
        raise


if __name__ == '__main__':
    main()
