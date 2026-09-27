"""One allocation, independent architecture subprocesses, no retry or training."""

import json
import os
import subprocess
import sys
import traceback
from . import attempt_004 as attempt
from .evidence import create_record, update_record
from .provenance_004 import require_submission, sha
from .records_003 import interrupt_file


def main():
    manifest=require_submission()
    attempt.require_unused()
    identity=dict(attempt_id='004',job_id=os.environ['SLURM_JOB_ID'],execution_commit=os.environ['RUN_COMMIT'],
        source_hash=manifest['source_hash'],plan_sha256=sha(attempt.PLAN))
    reservation=json.loads(attempt.SUBMISSION.read_text())
    for key in ('execution_commit','source_hash'):
        if reservation[key]!=identity[key]:raise ValueError('Reservation mismatch: '+key)
    if reservation.get('job_id',identity['job_id'])!=identity['job_id']:raise ValueError('Wrong job owner')
    create_record(attempt.LOCK,identity)
    state=dict(**identity,status='RUNNING',architectures={},training_authorized=False,scientific_fits=0,TRAIN=0,VALID=0,TEST=0)
    create_record(attempt.PIPELINE,state)
    try:
        for arch in ('SISO','MIMO'):
            require_submission()
            state['stage']=arch
            update_record(attempt.PIPELINE,state)
            path=attempt.evidence(arch)
            with (attempt.LOGS/f'{arch.lower()}_stdout.log').open('x') as out, (attempt.LOGS/f'{arch.lower()}_stderr.log').open('x') as err:
                try:
                    code=subprocess.run([sys.executable,'-m','experiments.mamba3_three_time.gpu_diagnostics_004',
                        '--architecture',arch,'--output',str(path)],stdout=out,stderr=err,check=False,timeout=2400).returncode
                except subprocess.TimeoutExpired:
                    code=124
                    err.write('Frozen 40-minute architecture budget exceeded; no retry.\n')
            interrupt_file(path)
            child=json.loads(path.read_text()) if path.exists() else dict(status='MISSING')
            state['architectures'][arch]=dict(exit_code=code,status=child['status'],evidence=str(path),
                coverage=child.get('coverage'),verdicts=child.get('verdicts'),
                failed_checks=[r['case_id'] for r in child.get('cases',[]) if not r['passed']])
            update_record(attempt.PIPELINE,state)
        state['status']='DIAGNOSTICS_COMPLETE' if all(v['status']=='DIAGNOSTICS_COMPLETE' and v['exit_code']==0
            for v in state['architectures'].values()) else 'INCOMPLETE_DIAGNOSTICS'
    except Exception:
        state.update(status='DIAGNOSTIC_ERROR',traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        update_record(attempt.PIPELINE,state)
        create_record(attempt.SUMMARY,state)
    print(json.dumps({k:v for k,v in state.items() if k!='architectures'},indent=2),flush=True)
    return 0 if state['status']=='DIAGNOSTICS_COMPLETE' else 1


if __name__=='__main__':raise SystemExit(main())
