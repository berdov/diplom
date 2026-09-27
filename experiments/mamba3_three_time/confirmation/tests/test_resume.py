"""Real JSON roundtrips, production pairing/resolution, no optimizer steps."""
import ast
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import torch
from experiments.mamba3_three_time.confirmation import config as c, provenance as p, state
from experiments.mamba3_three_time.confirmation import resume_config as r, resume_provenance as q, resume_report
from experiments.mamba3_three_time.confirmation.tests.test_no_git import no_git


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def representative(mode, groups=1):
    optimizer = torch.optim.Adam([{'params':[torch.nn.Parameter(torch.ones(1))], 'lr':.001+i*.001} for i in range(groups)])
    row = q.read(r.ARCHIVE/'runs'/f'mamba3_three_time_confirm_siso_{mode}_seed2027_001.json')
    row['optimizer_settings'] = [{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups]
    row['first_train_batch_sha256'] = 'same-consumed-batch'
    return row, optimizer


class OptimizerRoundtrip(unittest.TestCase):
    def test_original_failure_and_production_before_after_fit(self):
        tree = ast.parse((r.ARCHIVE/'state.py').read_text())
        node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='paired')
        scope = {}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<historical paired>','exec'),scope)
        a, optimizer = representative('dual'); b, _ = representative('triple')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'complete_record.json'
            write(path,a); loaded = q.read(path)
            with self.assertRaisesRegex(ValueError,'optimizer_settings'): scope['paired'](loaded,b)
            for before,after in ((loaded,b),(b,loaded)):
                # Reversing the records also reverses the fixed calibrator mapping.
                if before['mode']=='triple':
                    before=copy.deepcopy(before);after=copy.deepcopy(after)
                    before['initial_calibrator_hashes']=a['initial_calibrator_hashes']
                    after['initial_calibrator_hashes']=b['initial_calibrator_hashes']
                state.paired(before,after)
                state.paired(before,after,first_batch=True)
            write(path,b); state.paired(loaded,q.read(path),first_batch=True)
            normalized=state.canonical_optimizer_settings(b['optimizer_settings'])
            self.assertIsInstance(normalized[0]['betas'],list)
            self.assertIsInstance(optimizer.param_groups[0]['betas'],tuple)
            self.assertEqual(optimizer.state,{})

    def test_real_parameter_differences_and_paths(self):
        a,_=representative('dual',2); b,_=representative('triple',2)
        changes={'lr':.002,'betas':[.9,.998],'eps':1e-7,'weight_decay':.1,
                 'amsgrad':True,'foreach':False,'fused':False,'capturable':True,'maximize':True}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'record.json';write(path,a);a=q.read(path)
            for key,value in changes.items():
                if key not in b['optimizer_settings'][0]: continue
                bad=copy.deepcopy(b);bad['optimizer_settings'][0][key]=value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError,r'optimizer_settings\[0\]\.'+key):
                    state.paired(a,bad,first_batch=True)
            for mutate in (lambda v:v.reverse(),lambda v:v.pop(),lambda v:v.append(copy.deepcopy(v[0])),
                           lambda v:v[0].pop('eps'),lambda v:v[0].update(betas=[.999,.9]),
                           lambda v:v[0].update(amsgrad=None),lambda v:v[0].update(amsgrad=0)):
                bad=copy.deepcopy(b);mutate(bad['optimizer_settings'])
                with self.assertRaisesRegex(ValueError,'optimizer_settings'): state.paired(a,bad)

    def test_invalid_types_nonfinite_and_other_pair_guards(self):
        a,_=representative('dual');b,_=representative('triple')
        for value in (float('nan'),float('inf'),-float('inf'),object(),{1,2},torch.tensor(.1)):
            bad=copy.deepcopy(b);bad['optimizer_settings'][0]['lr']=value
            with self.assertRaisesRegex(ValueError,'optimizer_settings'): state.paired(a,bad)
        for key in ('initial_backbone_sha256','rng_components','rng_before_fit_sha256','first_train_batch_sha256'):
            bad=copy.deepcopy(b);bad[key]='changed'
            with self.assertRaisesRegex(ValueError,key):state.paired(a,bad,first_batch=True)
        self.assertFalse(torch.cuda.is_initialized())


class ResumeFixtures(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        root=Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        plan=q.read(r.PLAN);here=root/c.HERE.relative_to(c.ROOT)
        for entry in plan['source_index']:
            path=root/entry['path']
            if entry['kind']=='NEW':
                row=dict(mode=entry['mode'],seed=entry['seed'],run_id=c.task(entry['mode'],entry['seed'])['run_id'],
                         job_id='new-job',execution_commit='new-commit',source_hash='new-hash',execution_attempt=3,
                         TEST='NOT_RUN',test_evaluation_count=0,status='NOT_RUN',scientific_fit_started=False)
            else:
                source=c.ROOT/(entry.get('archive_path') or entry['path'])
                row=q.read(source)
                if entry['kind']=='REUSED_COMPLETED':
                    row.update(checkpoint_path=str(root/entry['checkpoint_path']),checkpoint_metadata_path=str(root/entry['metadata_path']))
                    cp=root/entry['checkpoint_path'];cp.parent.mkdir(parents=True,exist_ok=True);cp.write_bytes(b'checkpoint fixture, never deserialized')
                    row['checkpoint_sha256']=entry['checkpoint_sha256']=p.sha(cp)
                    meta=q.read(r.ARCHIVE/'slurm_logs'/row['run_id']/'checkpoints/best_metadata.json')
                    meta['checkpoint_sha256']=row['checkpoint_sha256']
                    write(root/entry['metadata_path'],meta);entry['metadata_sha256']=p.sha(root/entry['metadata_path'])
            write(path,row)
            if entry['kind']!='NEW': entry['sha256']=p.sha(path)
        write(here/'resume_plan_003.json',plan)
        self.stack.enter_context(patch.multiple(c,ROOT=root))
        self.stack.enter_context(patch.multiple(r,ROOT=root,HERE=here,PLAN=here/'resume_plan_003.json',
            LOGS=here/'slurm_logs/attempt_003',RUNS=here/'runs/attempt_003',LOCK=here/'slurm_logs/attempt_003/pipeline.lock',
            SUMMARY=here/'runs/attempt_003/confirmation_summary.json',SUBMISSION=here/'slurm_logs/attempt_003/submission_003.json',LOGIN=here/'slurm_logs/attempt_003/login.json'))
        self.root=root;self.here=here
        self.base=dict(job_id='new-job',execution_commit='new-commit',source_hash='new-hash',execution_attempt=3,TEST='NOT_RUN',test_evaluation_count=0)

    def test_exact_queue_parent_acceptance_partial_summary_and_old_files(self):
        self.assertEqual([(t['mode'],t['seed']) for t in r.plan()['tasks']],list(r.ORDER))
        with self.assertRaises(ValueError):r.execution_task('dual',2027)
        old=self.here/'runs/old_fail.json';write(old,{'status':'FAIL'});before=old.read_bytes()
        row=q.resolve('dual',2027,self.base)
        self.assertEqual(row['job_id'],'4355052')
        value=resume_report.collect(self.base)
        self.assertEqual(value['scientific_fits_this_attempt'],0)
        self.assertEqual(value['total_completed_confirmation_fits'],1)
        self.assertEqual(value['summaries']['all_five_pairs']['full']['n_expected_runs'],10)
        self.assertEqual(value['summaries']['new_four_pairs']['full']['n_expected_runs'],8)
        self.assertEqual(value['status'],'INCOMPLETE')
        self.assertEqual(old.read_bytes(),before)

    def test_cross_attempt_pair_before_and_after_real_json_save(self):
        dual=q.resolve('dual',2027,self.base)
        triple=copy.deepcopy(dual)
        triple.update(self.base,mode='triple',run_id=c.task('triple',2027)['run_id'],parameter_count=c.COUNTS['triple'])
        triple['initial_calibrator_hashes']={'decay':dual['initial_calibrator_hashes']['decay'],
            'write':dual['initial_calibrator_hashes']['scan'],'phase':dual['initial_calibrator_hashes']['scan']}
        triple['optimizer_settings'][0]['betas']=tuple(triple['optimizer_settings'][0]['betas'])
        state.paired(dual,triple)
        write(r.paths('triple',2027)['result'],triple)
        loaded=q.resolve('triple',2027,self.base,check_bytes=False)
        state.paired(dual,loaded,first_batch=True)
        for k in ('initial_backbone_sha256','rng_components'):
            bad=copy.deepcopy(loaded);bad[k]='wrong'
            with self.assertRaises(ValueError):state.paired(dual,bad)
        loaded['optimizer_settings'][0]['lr']=.01
        with self.assertRaises(ValueError):state.paired(dual,loaded)

    def test_wrong_parent_json_checkpoint_and_unrelated_pass(self):
        entry=next(x for x in r.plan()['source_index'] if x['kind']=='REUSED_COMPLETED')
        cp=self.root/entry['checkpoint_path'];original=cp.read_bytes();cp.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Checkpoint SHA'):q.resolve('dual',2027,self.base)
        cp.write_bytes(original)
        path=self.root/entry['path'];row=q.read(path);row['job_id']='random PASS';write(path,row)
        with self.assertRaisesRegex(ValueError,'Unauthorized inherited'):q.resolve('dual',2027,self.base)
        new=r.paths('triple',2027)['result'];row=q.read(new);row.update(status='PASS',job_id='unrelated');write(new,row)
        with self.assertRaisesRegex(ValueError,'ownership'):q.resolve('triple',2027,self.base)

    def test_duplicate_reservation_and_owned_artifacts(self):
        for t in r.plan()['tasks']:
            r.paths(t['mode'],t['seed'])['result'].unlink()
        r.unused(include_submission=True)
        p.create_record(r.SUBMISSION,{'status':'RESERVED'})
        with self.assertRaises(FileExistsError):r.unused(include_submission=True)
        with self.assertRaises(FileExistsError):p.create_record(r.SUBMISSION,{'status':'RESERVED'})
        self.assertEqual(q.read(r.SUBMISSION),{'status':'RESERVED'})


@unittest.skipUnless(importlib.util.find_spec('recbole') and str(c.ROOT)=='/home/daryumin/iberdov/diplom', 'Existing cluster runtime required')
class ResumeNoGit(unittest.TestCase):
    def test_real_runtime_and_launcher_without_git(self):
        m=p.verify();commit='f'*40
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            login=dict(**q.bindings(commit,m),status='PASS',parent_git_objects_verified=True,
                       tracked_sources_clean=True,published_branch_commit=commit)
            p.create_record(directory/'login.json',login)
            submission=dict(**q.bindings(commit,m),status='RESERVED',max_scientific_fits=7,
                            new_diagnostic_optimizer_steps=0,login_verification_sha256=p.sha(directory/'login.json'))
            p.create_record(directory/'submission.json',submission);p.create_record(directory/'test_fixture.json',{'test':True})
            env=dict(os.environ,RUN_COMMIT=commit,SLURM_JOB_ID='12345',EXPECTED_STUDY_HASH=m['source_hash'],
                     EXPECTED_CORE_HASH=p.CORE,PYTHONPATH=str(c.ROOT),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
            with patch.dict(os.environ,env),no_git() as attempted:
                self.assertEqual(q.identity(submission_path=directory/'submission.json',login_path=directory/'login.json')['execution_attempt'],3)
                self.assertFalse(attempted)
            response=subprocess.run(['/bin/bash',str(c.ROOT/'slurm/mamba3_three_time_confirmation_resume.sh'),
                '--runtime-preflight-only',str(directory)],env=dict(env,PATH=''),capture_output=True,text=True,timeout=180)
            self.assertEqual(response.returncode,0,response.stdout+response.stderr)
            self.assertFalse(torch.cuda.is_initialized())
            self.assertEqual(set(x.name for x in directory.iterdir()),{'login.json','submission.json','test_fixture.json'})


if __name__=='__main__':
    unittest.main()
