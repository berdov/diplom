import ast
import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from experiments.mamba3_mimo_time.confirmation import config as c, provenance as p, pipeline, report, submit
from experiments.mamba3_mimo_time.confirmation.records import create, read, sha, digest
from experiments.mamba3_mimo_time.process_env import child_environment, visibility

TORCH = importlib.util.find_spec('torch') is not None


@contextmanager
def isolated():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        logs, runs = root/'slurm_logs', root/'runs'
        with patch.multiple(c,HERE=root,LOGS=logs,RUNS=runs,LOGIN=logs/'login_verification_001.json',
                            RESERVATION=logs/'reservation_001.json',SUBMISSION=logs/'submission_001.json',
                            PIPELINE=logs/'pipeline_status.json',INHERITED=runs/'inherited_admission_001.json',
                            SUMMARY=runs/'confirmation_summary.json'):
            yield root


def sample(seed,mode,score=None):
    row = copy.deepcopy(read(c.pilot_path(mode)))
    row.update(seed=seed,run_id=c.paths(mode,seed)['run_id'])
    if score is not None:
        row['history'] = [copy.deepcopy(row['history'][row['best_epoch']])]
        h = row['history'][0]
        h.update(epoch=0,valid_ndcg10=score)
        h['valid_metrics']['ndcg@10'] = score
        row.update(actual_epochs=1,best_epoch=0,best_valid_score=score,best_valid_metrics=h['valid_metrics'],best_diagnostics=h['diagnostics'])
    return row


class FrozenTests(unittest.TestCase):
    def test_exact_plan_and_twelve_fixed_ids(self):
        self.assertEqual(c.plan(),c.expected_plan())
        self.assertEqual([(r['seed'],r['mode']) for r in c.tasks()],[(s,m) for s in (2027,2028,2029,2030) for m in ('base','dual','triple')])
        self.assertEqual(len({r['run_id'] for r in c.tasks()}),12)
        for seed in (2026,2031):
            with self.assertRaises(ValueError):c.paths('dual',seed)

    def test_only_seed_paths_change_in_saved_config(self):
        for seed in (2026,*c.SEEDS):
            for mode in c.MODES:
                actual = c.settings(mode,seed,checkpoint_dir='/temporary/checkpoints')
                expected = copy.deepcopy(read(c.pilot_path(mode))['config'])
                expected.update(seed=seed,checkpoint_dir='/temporary/checkpoints')
                self.assertEqual(actual,expected)
                cpu = c.settings(mode,seed,'cpu','/temporary/checkpoints')
                self.assertEqual((cpu['device'],cpu['use_gpu'],cpu['gpu_id']),('cpu',False,''))

    def test_inherited_full_leaf_coverage_and_policy(self):
        evidence = p.inherited()
        self.assertEqual((evidence['cases'],evidence['required_checks']),(45,2342))
        self.assertEqual(sha(c.POLICY),c.POLICY_SHA)
        old = read(c.PILOT_ROOT/'source_manifest.json')
        self.assertEqual(digest(old['files']),c.PILOT_SOURCE)
        p.check_files(c.ROOT,old)

    def test_hook_lifecycle_from_real_preserved_production_evidence(self):
        rows = [r for r in read(c.ADMISSION)['cases'] if r['case_id'].startswith('prefix_')]
        self.assertEqual(len(rows),12)
        phases = [v for row in rows for v in row['prefix_phases'].values()]
        self.assertEqual(len(phases),20)
        for phase in phases:
            self.assertEqual(phase['capture_count'],1)
            self.assertTrue(phase['hook_removed'])
            events = phase['stages']
            removed = next(i for i,e in enumerate(events) if e['stage']=='hook_removed')
            after = [(i,e) for i,e in enumerate(events) if e['stage']=='intervention_completed']
            self.assertEqual(len(after),2)
            self.assertTrue(all(i>removed and not e['grad_enabled'] and not e['output_requires_grad'] for i,e in after))
        manifest = read(c.PILOT_ROOT/'source_manifest.json')
        for name in ('prefix_checks.py','trainer.py','runner.py','config.py'):
            file = c.PILOT_ROOT/name
            self.assertEqual(sha(file),manifest['files'][str(file.relative_to(c.ROOT))])

    def test_runner_setup_order_matches_pilot(self):
        def calls(path):
            node = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='train')
            selected = []
            wanted = {'Config','init_seed','init_logger','verify_protocol','PreciseHistoryDataset','build','verify_history_stats','TrainDataLoader','FullSortEvalDataLoader','ThreeTimeMamba3Rec','trainer_class','cls','initial','fit'}
            for n in ast.walk(node):
                if isinstance(n,ast.Call):
                    name = n.func.id if isinstance(n.func,ast.Name) else n.func.attr if isinstance(n.func,ast.Attribute) else ''
                    if name in wanted:selected.append((n.lineno,name))
            return [name for _,name in sorted(selected)]
        self.assertEqual(calls(c.HERE/'runner.py'),calls(c.PILOT_ROOT/'runner.py'))
        source = (c.HERE/'runner.py').read_text()
        self.assertIn('del reserved',source)
        self.assertNotIn('torch.load',source)
        self.assertNotIn('seed=2026',source)


class GuardTests(unittest.TestCase):
    def test_hash_and_manifest_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            create(root/'data.json',{'a':1})
            files = {'data.json':sha(root/'data.json')}
            manifest = dict(files=files,source_hash=digest(files))
            p.check_files(root,manifest)
            with self.assertRaises(ValueError):p.check_files(root,dict(manifest,source_hash='wrong'))
            (root/'data.json').write_text('{}')
            with self.assertRaises(ValueError):p.check_files(root,manifest)

    def test_evidence_tampering_blocks_inheritance(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d)/'admission.json'
            row = read(c.ADMISSION)
            row['cases'][0]['checks']['initialization']['passed'] = False
            create(file,row)
            with patch.object(c,'ADMISSION',file),self.assertRaises(ValueError):p.inherited()
            with patch.object(c,'ADMISSION',file),patch.object(c,'ADMISSION_SHA',sha(file)),self.assertRaisesRegex(ValueError,'admission'):
                p.inherited()

    def test_serialized_reservation_tampering(self):
        expected = dict(execution_commit='a'*40,source_hash='source',source_manifest_sha256='manifest',policy_sha256=c.POLICY_SHA)
        login = dict(expected,status='PASS',tracked_clean=True,source_blobs_verified=True,published_commit='a'*40)
        reserve = dict(expected,status='RESERVED',login_sha256='login',max_scientific_fits=12,tasks=c.tasks(),jobs_requested=1,token='b'*32)
        with tempfile.TemporaryDirectory() as d:
            create(Path(d)/'login',login);create(Path(d)/'reserve',reserve)
            login,reserve = read(Path(d)/'login'),read(Path(d)/'reserve')
            p.validate_ownership(login,reserve,'login',expected,'999')
            for key,value in [('source_hash','bad'),('source_manifest_sha256','bad'),('policy_sha256','bad'),('execution_commit','c'*40),('max_scientific_fits',13),('tasks',[]),('jobs_requested',2),('token','bad')]:
                with self.assertRaises(ValueError):p.validate_ownership(login,dict(reserve,**{key:value}),'login',expected,'999')
            with self.assertRaises(ValueError):p.validate_ownership(login,reserve,'login',expected,'999',dict(token='b'*32,job_id='998'))

    def test_atomic_create_and_output_guards(self):
        with isolated():
            c.unused()
            create(c.paths('base',2027)['lock'],{'owner':'first'})
            with self.assertRaises(FileExistsError):c.unused()
            with self.assertRaises(FileExistsError):create(c.paths('base',2027)['lock'],{'owner':'second'})
            self.assertEqual(read(c.paths('base',2027)['lock']),{'owner':'first'})

    def test_submit_exactly_once_and_ambiguous_no_retry(self):
        for code,stdout in ((0,'999\n'),(1,'')):
            with isolated() as root:
                create(root/'source_manifest.json',{'files':{},'source_hash':'source'})
                create(root/'study_plan.json',{})
                expected = p.bindings('a'*40,dict(source_hash='source'))
                login = dict(expected,status='PASS',tracked_clean=True,published_commit='a'*40,source_blobs_verified=True)
                for name in ('cpu_tests_001.json','login_preflight_001.json','no_git_preflight_001.json','ownership_001.json'):
                    create(c.LOGS/name,dict(status='PASS',source_hash='source',execution_commit='a'*40))
                def fake(*args,**kwargs):
                    self.assertTrue(c.RESERVATION.is_file())
                    self.assertTrue(c.LOGIN.is_file())
                    self.assertEqual(args[0][0],'sbatch')
                    return subprocess.CompletedProcess(args[0],code,stdout,'')
                canonical = Path('/home/daryumin/iberdov/diplom')
                with patch.object(c,'ROOT',canonical),patch.object(c,'LAUNCHER',canonical/'slurm/mamba3_mimo_confirmation.sh'),patch.object(submit,'login_verify',return_value=login),patch.object(submit,'verify',return_value=dict(source_hash='source')),patch.object(submit.subprocess,'run',side_effect=fake) as run:
                    if code:
                        with self.assertRaisesRegex(RuntimeError,'do not repeat'):submit.main()
                    else:submit.main()
                    with self.assertRaises(FileExistsError):submit.main()
                    self.assertEqual(run.call_count,1)
                self.assertEqual(read(c.RESERVATION)['max_scientific_fits'],12)
                self.assertEqual(read(c.SUBMISSION)['status'],'SUBMITTED' if code==0 else 'SUBMISSION_UNKNOWN_NO_RETRY')

    def test_production_subprocess_cpu_mask_and_gpu_inheritance(self):
        probe = "import os,json;print(json.dumps({'present':'CUDA_VISIBLE_DEVICES' in os.environ,'value':os.environ.get('CUDA_VISIBLE_DEVICES')}))"
        for value in ('0','GPU-marker',None):
            env = dict(os.environ)
            if value is None:env.pop('CUDA_VISIBLE_DEVICES',None)
            else:env['CUDA_VISIBLE_DEVICES']=value
            with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,env,clear=True):
                for stage in ('preflight',*c.MODES):
                    target = Path(d)/stage
                    pipeline.child(stage,'timeit',['-n','1','-r','1',probe],time.time()+30,target)
                    actual = json.loads((target/'stdout.log').read_text().splitlines()[0])
                    self.assertEqual(actual,dict(present=True,value='') if stage=='preflight' else visibility(env))
                    self.assertEqual(dict(os.environ),env)


class AggregateTests(unittest.TestCase):
    def pilot(self):
        return {(2026,m):read(c.pilot_path(m)) for m in c.MODES}

    def test_missing_are_null_and_no_pilot_as_confirmation(self):
        summary = report.summarize({},self.pilot())
        self.assertEqual(summary['status'],'INCOMPLETE')
        a,b = summary['confirmatory'],summary['with_exploratory_pilot']
        self.assertEqual((a['n_available'],a['n_expected']),(0,4))
        self.assertEqual((b['n_available'],b['n_expected']),(1,5))
        self.assertIsNone(a['model_stats_same_complete_triples']['base']['mean'])
        self.assertIsNone(a['contrasts'][0]['delta']['sample_std'])
        self.assertIn('неполное',report.markdown(summary))

    def test_negative_zero_positive_pairs_ddof1_and_incomplete_windows(self):
        records = {}
        for seed,delta in zip(c.SEEDS,[-.002,-.001,0,.003]):
            for mode,score in [('base',.05),('dual',.06),('triple',.06+delta)]:
                records[(seed,mode)] = sample(seed,mode,score)
        a = report.summarize(records,self.pilot())['confirmatory']
        r = a['contrasts'][0]
        self.assertEqual((r['positive'],r['negative'],r['zero']),(1,2,1))
        self.assertAlmostEqual(r['delta']['mean'],0)
        self.assertAlmostEqual(r['delta']['sample_std'],(14e-6/3)**.5)
        self.assertEqual(r['delta']['ddof'],1)
        self.assertTrue(all(not row['first27_complete'] and row['first27_observed_epochs']==1 for row in a['rows']))

    def test_means_always_use_matching_seed_sets(self):
        records = {(2027,'base'):sample(2027,'base'),(2028,'dual'):sample(2028,'dual'),(2028,'triple'):sample(2028,'triple')}
        a = report.summarize(records,self.pilot())['confirmatory']
        self.assertEqual(a['complete_triple_seeds'],[])
        self.assertEqual(a['contrasts'][0]['seeds'],[2028])
        self.assertEqual(a['contrasts'][1]['seeds'],[])
        self.assertEqual(a['contrasts'][0]['status'],'INCOMPLETE_SUBSET')

    def test_bad_history_best_and_test_rejected(self):
        for field,value in [('TEST','RUN'),('test_evaluation_count',1),('best_epoch',0),('actual_epochs',99),('run_id','wrong')]:
            row = sample(2027,'dual');row[field]=value
            with self.assertRaises(ValueError):report.summarize({(2027,'dual'):row},self.pilot())


class PipelineTests(unittest.TestCase):
    def exercise(self,fail_at=None,timeout=False):
        planned = c.plan()
        base = dict(execution_commit='a'*40,source_hash='test-source',job_id='999',
                    reservation_token='b'*32,TEST='NOT_RUN',test_evaluation_count=0)
        calls = []
        with isolated(),patch.dict(os.environ,{'SLURM_JOB_END_TIME':str(time.time()+(1 if timeout else 28800))}):
            def child(stage,module,args,deadline,directory):
                mode,seed = args[1],int(args[3])
                calls.append((seed,mode))
                if len(calls)==fail_at:
                    raise RuntimeError('synthetic technical failure')
                row = sample(seed,mode,.03 if mode=='triple' else .06)
                row.update(base)
                paths = c.paths(mode,seed)
                paths['checkpoint'].parent.mkdir(parents=True,exist_ok=True)
                # Deliberately not a model: tests only exercise file-integrity plumbing.
                paths['checkpoint'].write_bytes(b'not a model; never deserialized')
                row['checkpoint_sha256'] = sha(paths['checkpoint'])
                create(paths['result'],row)
                create(paths['metadata'],dict(metrics=row['best_valid_metrics'],epoch=row['best_epoch'],
                       run_id=row['run_id'],execution_commit=row['execution_commit'],checkpoint_sha256=row['checkpoint_sha256']))
            with patch.object(c,'plan',return_value=planned),patch.object(pipeline,'identity',return_value=base),patch.object(pipeline,'inherited',return_value={'status':'INHERITED_PASS'}),patch.object(pipeline,'runtime',return_value={}),patch.object(pipeline,'require_inherited',return_value='hash'),patch.object(pipeline,'child',side_effect=child):
                import signal
                handler = signal.getsignal(signal.SIGTERM)
                try:
                    code = pipeline.main()
                finally:
                    signal.signal(signal.SIGTERM,handler)
            status,summary = read(c.PIPELINE),read(c.SUMMARY)
            self.assertTrue(c.SUMMARY.with_suffix('.md').is_file())
        return code,status,summary,calls

    def test_low_metric_never_stops_twelve_fresh_processes(self):
        code,status,summary,calls = self.exercise()
        self.assertEqual(code,0)
        self.assertEqual(calls,[(s,m) for s in c.SEEDS for m in c.MODES])
        self.assertEqual(summary['scientific_fits_completed'],12)
        self.assertEqual(summary['confirmatory']['contrasts'][0]['negative'],4)

    def test_technical_failure_stops_and_saves_partial(self):
        code,status,summary,calls = self.exercise(fail_at=3)
        self.assertEqual(code,1)
        self.assertEqual(len(calls),3)
        self.assertEqual(summary['scientific_fits_completed'],2)
        self.assertEqual(status['status'],'FAIL')
        self.assertEqual(summary['status'],'INCOMPLETE')
        self.assertIn('synthetic technical failure',status['traceback'])

    def test_deadline_prevents_start_without_shortening_settings(self):
        code,status,summary,calls = self.exercise(timeout=True)
        self.assertEqual(code,1)
        self.assertFalse(calls)
        self.assertEqual(status['status'],'INCOMPLETE')
        self.assertEqual(summary['scientific_fits_started'],0)


@unittest.skipUnless(TORCH,'Frozen cluster environment required; no packages installed locally')
class RuntimeCPUTests(unittest.TestCase):
    def test_real_adam_json_and_base_aware_pairing(self):
        import torch
        from experiments.mamba3_three_time.confirmation.state import canonical_optimizer_settings,compare_optimizer_settings
        from experiments.mamba3_mimo_time.state import paired
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(2))],lr=.001)
        raw = [{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups]
        with tempfile.TemporaryDirectory() as d:
            create(Path(d)/'adam.json',canonical_optimizer_settings(raw))
            compare_optimizer_settings(raw,read(Path(d)/'adam.json'))
        bad = copy.deepcopy(raw);bad[0]['lr']=.02
        with self.assertRaises(ValueError):compare_optimizer_settings(raw,bad)
        rows = [read(c.pilot_path(m)) for m in c.MODES]
        for a,b in ((0,1),(0,2),(1,2)):paired(rows[a],rows[b],first_batch=True)
        rows[2]['initial_calibrator_hashes']['phase']='wrong'
        with self.assertRaises(ValueError):paired(rows[1],rows[2])

    def test_trainer_and_safe_checkpoint_paths_inherited(self):
        from recbole.trainer import Trainer
        from experiments.mamba3_mimo_time.trainer import trainer_class
        cls = trainer_class(Trainer,object(),{}, {})
        self.assertIs(cls.fit,Trainer.fit)
        self.assertEqual(cls._save_checkpoint.__module__,'experiments.mamba3_context_time.trainer')
        self.assertEqual(cls.evaluate.__module__,'experiments.mamba3_context_time.trainer')
        self.assertEqual(cls._valid_epoch.__module__,'experiments.mamba3_mimo_time.trainer')

    def test_real_launcher_cpu_preflight_no_git(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d)/'evidence.json'
            env = dict(os.environ,PATH='/nonexistent',CUDA_VISIBLE_DEVICES='GPU-parent',REPO_ROOT=str(c.ROOT))
            proc = subprocess.run(['/bin/bash',str(c.LAUNCHER),'--preflight-only','--evidence',str(target)],cwd=c.ROOT,env=env,capture_output=True,text=True,timeout=600)
            self.assertEqual(proc.returncode,0,proc.stderr)
            r = read(target)
            self.assertEqual(r['status'],'PASS')
            self.assertFalse(r['final_cuda_initialized'])
            self.assertEqual(len(r['initialization']['rows']),15)
            self.assertEqual(r['model_forward_calls'],0)
            for stage in r['stages']:
                self.assertEqual(stage['cuda_mask'],'')
                self.assertFalse(stage['cuda_initialized'])
                if 'gpu_id' in stage:self.assertEqual(stage['gpu_id'],'')


if __name__ == '__main__':
    unittest.main()
