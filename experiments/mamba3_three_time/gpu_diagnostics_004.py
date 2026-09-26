"""Attempt004 synthetic diagnostics with orthogonal, non-authorizing verdicts."""

import argparse
import json
import os
import traceback
import torch
from . import attempt_004 as attempt, diagnostics_004 as diagnostics, mimo_diagnostics_004 as mimo
from .backends import selected
from .gpu_checks_003 import suite_factories
from .gpu_checks_002 import expected_suite_names
from .mimo_checks_003 import gate
from .records_003 import Registry
from .evidence import create_record, update_record
from .provenance_004 import require_submission, upstream, sha, CORE, PIN


def specs(arch,plan):
    rows=[]
    if arch=='SISO':
        rows.append(('trace_smoke',('trace',)))
        rows += [(f'drift_{m}_{t}_L{n}',('drift',m,t=='train',n)) for m,t,n in plan['drift_fixtures']]
        rows += [(name,('suite',name)) for name in expected_suite_names(arch,plan) if not name.startswith('G_')]
    else:
        rows += [(f'official_support_L{n}',('gate',n,n==16)) for n in (16,17,50)]
        rows += [(f'forensics_L{n}_{"tied" if tied else "triple"}_{"D_only" if only else "original"}',
                  ('mimo',n,tied,only)) for n,tied,only in ((15,False,False),(50,True,False),(7,True,True))]
    fixtures=plan['prefix_fixtures'] if arch=='SISO' else plan['mimo_prefix_fixtures']
    rows += [(f'causal_{m}_L{n}_P{p}_x{x}',('causal',m,n,p,x)) for m in ('base','dual','triple')
             for n,p in fixtures for x in plan['loss_multipliers']]
    return [(f'{arch}/upstream/{name}',spec) for name,spec in rows]


def aggregate(rows):
    if not rows:return 'UNKNOWN'
    return 'PASS' if all(r.get('passed') is True for r in rows) else 'FAIL'


def verdicts(result):
    rows=result.get('cases',[])
    suites=[r for r in rows if r.get('category')=='suite']
    parity=[r for r in suites if r['name'].startswith(('A_','B_','C_','F_official','H_'))]
    reference=[r for r in suites if 'reference' in r['name'] or 'three_path' in r['name']]
    causal=[r for r in rows if r.get('category')=='causal']
    forward=[v for r in causal for k,v in r.get('checks',{}).items() if 'suffix_invariance' in k or 'cross_user' in k]
    residual={r['case_id']:{name:v['report']['embedding_output_gradient'] for name,v in r['variants'].items()} for r in causal}
    legacy={r['case_id']:r['legacy_exact_zero'] for r in causal}
    mimo_rows=[r for r in rows if r.get('category')=='mimo']
    mimo_parity=[v for r in mimo_rows for common in r['common_vjp'] for v in common.get('split_vs_official',{}).values()]
    mimo_accuracy=[v for r in mimo_rows for common in r['common_vjp'] for b in common['backends'].values() for v in b['comparisons'].values()]
    mimo_outputs=[b['output_discrepancy'] for r in mimo_rows for common in r['common_vjp'] for b in common['backends'].values()]
    mimo_answers=[dict(case_id=r['case_id'],D_oracle={c['kind']:{b:{k:v['passed'] for k,v in m['D_oracle_comparisons'].items()}
        for b,m in c['backends'].items()} for c in r['common_vjp']},
        common_VJP={c['kind']:{b:aggregate(list(m['comparisons'].values())) for b,m in c['backends'].items()} for c in r['common_vjp']},
        output_accuracy={c['kind']:{b:m['output_discrepancy']['passed'] for b,m in c['backends'].items()} for c in r['common_vjp']},
        loss_error_explained={b:m['loss_diagnostics']['bound_explains_fp64_delta'] for b,m in r['legacy'].items()},
        legacy_reference={b:aggregate(list(m['legacy_comparisons'].values())) for b,m in r['legacy'].items()},
        new_split_error='SEE_COMMON_VJP; triple has no direct official three-slot oracle') for r in mimo_rows]
    drift=[dict(case_id=r['case_id'],metadata_passed=r['passed'],variants={k:dict(
        first_measured_divergence=v['first_measured_divergence'],
        failed_checks=[n for n,c in v['checks'].items() if not c['passed']],
        unique_mathematical_cause=v['unique_mathematical_cause']) for k,v in r['variants'].items()})
        for r in rows if r.get('category')=='drift' and 'variants' in r]
    return dict(instrumentation=aggregate([r for r in rows if r.get('category')=='trace']),
        implementation_parity=aggregate(parity if parity else mimo_parity),independent_derivative_checks=aggregate(reference if reference else mimo_accuracy),
        reference_accuracy=aggregate(reference if reference else mimo_outputs+mimo_accuracy),
        forward_prefix_invariance=aggregate(forward),backward_numerical_residual=residual,
        legacy_exact_zero=legacy,stable_adt_drift=drift,
        mimo_forensics=mimo_answers,training_authorized=False,
        acceptance_proposal='Keep structural parity and forward causality unchanged. Keep old exact-zero and '
            'reference failures visible. Consider reporting near-zero loss using output-error propagation '
            'and component magnitudes rather than only net loss; no new acceptance threshold is authorized. '
            'Missing metadata/oracle and calibration failures remain UNKNOWN/INCONCLUSIVE.')


def run(arch,path):
    manifest=require_submission()
    plan=json.loads(attempt.PLAN.read_text())
    result=dict(attempt_id='004',architecture=arch,status='RUNNING',backend='upstream',
        execution_commit=os.environ['RUN_COMMIT'],job_id=os.environ['SLURM_JOB_ID'],
        source_hash=manifest['source_hash'],core_hash=CORE,pinned_commit=PIN,
        plan=plan,plan_sha256=sha(attempt.PLAN),source_manifest_sha256=sha(attempt.SOURCE_MANIFEST),
        training_authorized=False,scientific_fits=0,TRAIN=0,VALID=0,TEST=0)
    create_record(path,result)
    save=lambda:update_record(path,result)
    schedule=specs(arch,plan)
    registry=Registry(result,save,[k for k,_ in schedule])
    try:
        result['upstream']=upstream()
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.cuda.set_device(0)
        torch.empty(1,device='cuda').zero_()
        torch.cuda.synchronize()
        torch.backends.cuda.matmul.allow_tf32=False
        result['gpu']=torch.cuda.get_device_name()
        if 'A100' not in result['gpu']:raise RuntimeError('Only A100 authorized')
        import importlib.metadata
        result['runtime']={n:importlib.metadata.version(n) for n in ('torch','triton','tilelang','mamba-ssm')}
        suite_functions=suite_factories(arch,plan) if arch=='SISO' else {}
        for case_id,spec in schedule:
            result['stage']=case_id
            save()
            def factory():
                kind=spec[0]
                if kind=='trace':row=diagnostics.trace_smoke()
                elif kind=='drift':row=diagnostics.drift(*spec[1:])
                elif kind=='suite':
                    row=suite_functions[spec[1]]()
                    if row['name']!=spec[1]:row.update(passed=False,status='INCONCLUSIVE',expected_name=spec[1])
                elif kind=='mimo':row=mimo.report(spec[1],spec[2],plan['reference'],spec[3])
                elif kind=='gate':row=gate(arch,spec[1],native=spec[2])
                else:row=diagnostics.causality(arch,*spec[1:])
                row.setdefault('backend','upstream')
                row.update(category=kind,training_authorized=False)
                return row
            with selected('upstream'):
                row=registry.run(case_id,factory)
            print(json.dumps(dict(case_id=case_id,status=row['status'])),flush=True)
            if spec[0]=='trace' and not row['passed']:
                result['status']='INSTRUMENTATION_FAIL'
                break
            if 'traceback' in row:
                result['status']='DIAGNOSTIC_ERROR'
                break
        else:result['status']='DIAGNOSTICS_COMPLETE'
    except Exception:
        result.update(status='DIAGNOSTIC_ERROR',traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        registry.close()
        result['verdicts']=verdicts(result)
        result['all_legacy_required_checks_passed']=all(r['passed'] for r in result['cases']) and not result['coverage']['missing']
        save()
    return 0 if result['status']=='DIAGNOSTICS_COMPLETE' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--architecture',choices=('SISO','MIMO'),required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    from pathlib import Path
    if Path(a.output).resolve()!=attempt.evidence(a.architecture).resolve():raise ValueError('Attempt004 path only')
    raise SystemExit(run(a.architecture,Path(a.output)))
