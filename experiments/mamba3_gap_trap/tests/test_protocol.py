"""Infrastructure fixtures never write into an experiment namespace."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from experiments.mamba3_gap_trap import config as c
from experiments.mamba3_gap_trap import pipeline, provenance, report, submit
from experiments.mamba3_gap_trap.process_env import child_environment
from experiments.mamba3_mimo_time.records import Registry, accepted_cases, create, read, sha


@contextmanager
def scratch():
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root=Path(directory)
        replacements=dict(ROOT=root,LOGS=root/'logs',RUNS=root/'runs',LAUNCHER=root/'launch.sh')
        for name in ('LOGIN','RESERVATION','SUBMISSION','PIPELINE'):
            replacements[name]=root/'logs'/(name.lower()+'.json')
        for name in ('INHERITED','GATE','SMOKE','SUMMARY'):
            replacements[name]=root/'runs'/(name.lower()+'.json')
        for name,value in replacements.items():stack.enter_context(patch.object(c,name,value))
        yield root


def successful(variant,score=.06,epochs=12):
    if score==0.:epochs=300
    best=epochs-12 if score else epochs-1
    h=[]
    for i in range(epochs):
        value=score if i<=best else score-.001
        metrics={f'{kind}@{k}':value if kind=='ndcg' else .1 for kind in ('hit','ndcg','recall') for k in (5,10,20,50)}
        h.append(dict(epoch=i,valid_ndcg10=value,valid_metrics=metrics,diagnostics={}))
    return dict(run_id=c.paths(variant)['run_id'],gap_trap_mode=variant,mode='dual',seed=2026,
                parameter_count=c.COUNTS[variant],TEST='NOT_RUN',test_evaluation_count=0,
                scientific_fit_started=True,status='PASS',history=h,actual_epochs=epochs,
                first27_complete=epochs>=27,first27_best_ndcg10=max(x['valid_ndcg10'] for x in h[:27]) if epochs>=27 else None,
                best_epoch=best,best_valid_score=score,best_valid_metrics=h[best]['valid_metrics'],best_diagnostics={})



class ProtocolTests(unittest.TestCase):
    def test_frozen_required_registry_is_accepted(self):
        specs=c.plan()['required_cases']
        rows=[dict(case_id=s['id'],status='PASS',required_keys=s['required_keys'],
                   checks={k:dict(passed=True) for k in s['required_keys']}) for s in specs]
        self.assertTrue(accepted_cases(rows,specs))
        self.assertEqual(c.plan()['max_scientific_fits'],len(c.MODES))
        self.assertEqual(c.plan()['max_jobs'],1)
        self.assertEqual(c.plan()['test_evaluations'],0)

    def test_child_isolation_before_import_and_parent_unchanged(self):
        parent=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-fixture')
        original=dict(parent)
        for stage in ('gate','smoke',*c.MODES):
            self.assertEqual(child_environment(stage,parent)['CUDA_VISIBLE_DEVICES'],'GPU-fixture')
        child=child_environment('preflight',parent)
        value=subprocess.check_output([sys.executable,'-B','-c',
            'import os,sys,json;print(json.dumps([os.environ["CUDA_VISIBLE_DEVICES"],"torch" in sys.modules]))'],env=child,text=True)
        self.assertEqual(json.loads(value),['',False])
        self.assertEqual(parent,original)
        with self.assertRaises(ValueError):child_environment('unexpected',parent)

    def test_registry_missing_negative_nonfinite_duplicate_and_partial(self):
        specs=[dict(id='one',required_keys=['a','b'])]
        good=[dict(case_id='one',status='PASS',required_keys=['a','b'],checks=dict(a=dict(passed=True),b=dict(passed=True)))]
        self.assertTrue(accepted_cases(good,specs))
        for damage in ('missing','negative','nonfinite','duplicate'):
            value=copy.deepcopy(good)
            if damage=='missing':del value[0]['checks']['a']
            elif damage=='negative':value[0]['checks']['a']['passed']=False
            elif damage=='nonfinite':value[0]['checks']['a']['error']=float('nan')
            else:value+=copy.deepcopy(value)
            self.assertFalse(accepted_cases(value,specs))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'gate.json';value={};create(path,value)
            registry=Registry(path,value,specs)
            def fail(save):
                save(dict(checks={'a':{'passed':False}},required_keys=['a']))
                raise AssertionError('intentional fixture')
            with self.assertRaises(RuntimeError):registry.run(specs[0],fail)
            saved=read(path)['cases'][0]
            self.assertEqual(saved['status'],'FAIL')
            self.assertFalse(saved['checks']['a']['passed'])
            self.assertEqual(saved['missing_keys'],['b'])

    def test_immutable_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'record.json';create(p,dict(owner='one'));before=sha(p)
            with self.assertRaises(FileExistsError):create(p,dict(owner='two'))
            self.assertEqual(sha(p),before)

    def test_ownership_rejects_drift(self):
        expected=dict(execution_commit='a'*40,source_hash='b'*64,TEST='NOT_RUN',test_evaluation_count=0)
        login=dict(expected,status='PASS',tracked_clean=True,source_blobs_verified=True,published_commit='a'*40)
        reservation=dict(expected,status='RESERVED',login_sha256='c'*64,token='d'*32,
                         max_scientific_fits=2,tasks=c.plan()['tasks'],jobs_requested=1)
        provenance.validate_ownership(login,reservation,'c'*64,expected,'123')
        for key,value in dict(source_hash='foreign',max_scientific_fits=4,tasks=[],jobs_requested=2,
                              status='RETRY',token='bad',login_sha256='other').items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                provenance.validate_ownership(login,dict(reservation,**{key:value}),'c'*64,expected,'123')
        with self.assertRaises(ValueError):
            provenance.validate_ownership(login,reservation,'c'*64,expected,'123',dict(job_id='124',token='d'*32))

    def test_one_submit_reservation_precedes_call_and_blocks_repeat(self):
        with scratch(), patch.object(submit,'bindings',return_value={'execution_commit':'a'*40}):
            def sbatch(*args,**kwargs):
                self.assertTrue(c.RESERVATION.exists())
                self.assertEqual(read(c.RESERVATION)['login_sha256'],sha(c.LOGIN))
                self.assertEqual(args[0][0:2],['sbatch','--parsable'])
                return subprocess.CompletedProcess(args[0],0,'12345;fixture\n','')
            with patch.object(submit.subprocess,'run',side_effect=sbatch) as call:
                result=submit.reserve_and_submit(dict(execution_commit='a'*40),dict(source_hash='b'*64))
                self.assertEqual(result['job_id'],'12345')
                self.assertEqual(read(c.SUBMISSION)['job_id'],'12345')
                with self.assertRaises(FileExistsError):
                    submit.reserve_and_submit(dict(execution_commit='a'*40),dict(source_hash='b'*64))
                self.assertEqual(call.call_count,1)

    def test_ambiguous_submit_is_durable_and_never_retried(self):
        with scratch(), patch.object(submit,'bindings',return_value={'execution_commit':'a'*40}):
            with patch.object(submit.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'uncertain','')) as call:
                with self.assertRaises(RuntimeError):
                    submit.reserve_and_submit(dict(execution_commit='a'*40),dict(source_hash='b'*64))
                self.assertEqual(read(c.SUBMISSION)['status'],'SUBMISSION_UNKNOWN_NO_RETRY')
                with self.assertRaises(FileExistsError):
                    submit.reserve_and_submit(dict(execution_commit='a'*40),dict(source_hash='b'*64))
                self.assertEqual(call.call_count,1)

    def test_summary_preserves_negative_and_incomplete_first27(self):
        records={'fixed_replay':successful('fixed_replay',.06,12),'gap_trap':successful('gap_trap',.055,27)}
        result=report.summarize(json.loads(json.dumps(records)))
        self.assertEqual(result['status'],'PASS')
        self.assertEqual([r['first27_complete'] for r in result['rows']],[False,True])
        self.assertLess(result['contrasts'][0]['delta'],0)
        records['fixed_replay']=successful('fixed_replay',0.)
        self.assertIsNone(report.summarize(records)['contrasts'][0]['relative_percent'])
        bad=successful('gap_trap');bad['best_epoch']=1
        self.assertEqual(report.summarize({'gap_trap':bad})['rows'][1]['status'],'FAIL')

    def test_partial_pass_and_metric_mismatch_rejected(self):
        r=successful('fixed_replay')
        r['history']=r['history'][:-1];r['actual_epochs']-=1
        with self.assertRaises(ValueError):report.validate_record(r,'fixed_replay')
        r=successful('gap_trap');r['history'][1]['valid_metrics']['ndcg@10']+=.01
        with self.assertRaises(ValueError):report.validate_record(r,'gap_trap')

    def test_bad_checkpoint_keeps_numeric_and_markdown_summary(self):
        with scratch():
            create(c.paths('fixed_replay')['result'],successful('fixed_replay'))
            value=report.write({})
            self.assertEqual(value['rows'][0]['status'],'FAIL')
            self.assertEqual(value['rows'][1]['status'],'NOT_RUN')
            self.assertTrue(c.SUMMARY.exists())
            self.assertTrue(c.SUMMARY.with_suffix('.md').exists())
            self.assertEqual(read(c.paths('fixed_replay')['result'])['status'],'PASS')

    def test_smoke_accepts_boundary_but_requires_gradient(self):
        rows=[]
        for variant in c.MODES:
            alpha=[] if variant=='fixed_replay' else c.plan()['alpha_keys']
            names=c.plan()['common_parameter_keys']+alpha
            rows.append(dict(gap_trap_mode=variant,status='PASS',alpha_values={n:dict(before=0.,after=0.) for n in alpha},
                             roundtrip_passed=True,roundtrip='weights_only=True',peak_allocated_bytes=1,peak_reserved_bytes=2,
                             steps=[dict(step=i,finite_loss=True,loss=1.,gradient_norms=dict.fromkeys(names,1.)) for i in range(3)]))
        good=dict(batch=2048,history_length=50,kernel_length=56,steps_per_mode=3,rows=rows)
        provenance.validate_smoke(good)
        for damage in ('missing','nonfinite','bounds','roundtrip','learning'):
            bad=copy.deepcopy(good);row=bad['rows'][1];key=c.plan()['alpha_keys'][0]
            if damage=='missing':row['steps'][0]['gradient_norms'].pop(key)
            elif damage=='nonfinite':row['steps'][0]['loss']=float('inf')
            elif damage=='bounds':row['alpha_values'][key]['after']=-.01
            elif damage=='roundtrip':row['roundtrip_passed']=False
            else:
                for step in row['steps']:step['gradient_norms'][key]=0.
            with self.subTest(damage=damage),self.assertRaises(ValueError):provenance.validate_smoke(bad)

    def test_pipeline_stops_and_keeps_not_run_records(self):
        for fail_stage in ('gate','smoke','fixed_replay','gap_trap'):
            with self.subTest(stage=fail_stage),scratch(),ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ,{},clear=False))
                for name,value in (('identity',{}),('inherited',{}),('runtime',{}),('require_stage','fixture')):
                    stack.enter_context(patch.object(pipeline,name,return_value=value))
                stack.enter_context(patch.object(pipeline.signal,'signal'))
                stages=[]
                def child(stage,*args):
                    stages.append(stage)
                    if stage==fail_stage:raise RuntimeError('intentional stage failure')
                    if stage in c.MODES:create(c.paths(stage)['result'],successful(stage))
                stack.enter_context(patch.object(pipeline,'child',side_effect=child))
                stack.enter_context(patch.object(report,'validate_checkpoint'))
                self.assertEqual(pipeline.main(),1)
                order=['gate','smoke',*c.MODES]
                self.assertEqual(stages,order[:order.index(fail_stage)+1])
                saved=read(c.SUMMARY)
                self.assertEqual(saved['status'],'INCOMPLETE')
                self.assertEqual(saved['test_evaluation_count'],0)
                self.assertTrue(c.SUMMARY.with_suffix('.md').exists())
                for variant in c.MODES[0 if fail_stage not in c.MODES else c.MODES.index(fail_stage):]:
                    self.assertEqual(read(c.paths(variant)['result'])['status'],'NOT_RUN')

    def test_content_hash_verifier_rejects_tampering_without_git(self):
        from experiments.mamba3_mimo_time.records import digest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'file.py';p.write_text('fixture\n')
            files={'file.py':sha(p)};manifest=dict(files=files,source_hash=digest(files))
            with patch.object(subprocess,'run',side_effect=AssertionError('No subprocess expected')), \
                 patch.object(subprocess,'check_output',side_effect=AssertionError('No Git expected')):
                provenance.check_files(root,manifest)
                p.write_text('changed\n')
                with self.assertRaises(ValueError):provenance.check_files(root,manifest)

    def test_startup_failure_is_durable(self):
        with scratch(),patch.dict(os.environ,{'SLURM_JOB_ID':'987'}),patch.object(pipeline,'identity',side_effect=ValueError('fixture')):
            self.assertEqual(pipeline.main(),1)
            row=read(c.LOGS/'startup_failure_987.json')
            self.assertEqual(row['status'],'FAIL')
            self.assertFalse(row['identity_verified'])
            self.assertEqual(row['scientific_fits_started'],0)

    def test_malformed_pass_blocks_next_fit(self):
        with scratch(),ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ,{},clear=False))
            for name,value in (('identity',{}),('inherited',{}),('runtime',{}),('require_stage','fixture')):
                stack.enter_context(patch.object(pipeline,name,return_value=value))
            stack.enter_context(patch.object(pipeline.signal,'signal'))
            stages=[]
            def child(stage,*args):
                stages.append(stage)
                if stage=='fixed_replay':
                    r=successful(stage);r['run_id']='wrong';create(c.paths(stage)['result'],r)
            stack.enter_context(patch.object(pipeline,'child',side_effect=child))
            self.assertEqual(pipeline.main(),1)
            self.assertEqual(stages,['gate','smoke','fixed_replay'])
            self.assertEqual(read(c.SUMMARY)['rows'][0]['status'],'FAIL')
            self.assertEqual(read(c.paths('gap_trap')['result'])['status'],'NOT_RUN')


if __name__=='__main__':unittest.main()
