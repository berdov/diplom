import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments.mamba3_mimo_time import config as c, pipeline
from experiments.mamba3_mimo_time.process_env import child_environment, visibility, CUDA_MASK
from experiments.mamba3_mimo_time.records import create, read, sha, digest
from experiments.mamba3_mimo_time.retry import verify_parent, parent_binding, POLICY_SHA, PLAN_SHA
from experiments.mamba3_mimo_time.provenance import validate_ownership, verify

TORCH = importlib.util.find_spec('torch') is not None


class EnvironmentTests(unittest.TestCase):
    def test_production_popen_masks_only_cpu_child(self):
        probe = "import os,json,sys; print(json.dumps({'present':'CUDA_VISIBLE_DEVICES' in os.environ,'value':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch_imported':'torch' in sys.modules}))"
        for value in ('0', 'GPU-test-uuid', 'GPU-first,GPU-second', None, ''):
            parent = dict(os.environ, RUN_COMMIT='test-commit', RESERVATION_TOKEN='test-token', EXPECTED_STUDY_HASH='test-hash')
            if value is None:
                parent.pop(CUDA_MASK, None)
            else:
                parent[CUDA_MASK] = value
            original = copy.deepcopy(parent)
            for stage in ('preflight','admission','smoke',*c.MODES):
                expected = dict(original)
                if stage == 'preflight':
                    expected[CUDA_MASK] = ''
                self.assertEqual(child_environment(stage, parent), expected)
                self.assertEqual(parent, original)
            with tempfile.TemporaryDirectory() as d, patch.object(c, 'LOGS', Path(d)), patch.dict(os.environ, parent, clear=True):
                for stage in ('preflight','admission','smoke',*c.MODES):
                    pipeline.child(stage, 'timeit', ['-n','1','-r','1',probe], time.time()+30)
                    row = json.loads((Path(d)/stage/'stdout.log').read_text().splitlines()[0])
                    expected = dict(present=True, value='') if stage == 'preflight' else visibility(parent)
                    self.assertEqual(row, dict(expected, torch_imported=False))
                    self.assertEqual(dict(os.environ), original)
                    saved = read(Path(d)/stage/'child_environment.json')
                    self.assertEqual(saved['parent_cuda_visibility'], visibility(parent))
                    self.assertNotIn('test-token', json.dumps(saved))

    def test_scientific_settings_and_gpu_branch_unchanged(self):
        old = read(c.PARENT_EVIDENCE/'scientific_settings.json')
        for mode in c.MODES:
            for device in (None, 'cuda'):
                actual = c.settings(mode, device)
                expected = copy.deepcopy(old[mode])
                expected['checkpoint_dir'] = str(c.paths(mode)['checkpoint'].parent)
                if device == 'cuda':
                    expected.update(device='cuda', use_gpu=True)
                self.assertEqual(actual, expected)
                self.assertEqual(actual['gpu_id'], 0)
            cpu = c.settings(mode, 'cpu')
            self.assertEqual((cpu['gpu_id'],cpu['device'],cpu['use_gpu']), ('','cpu',False))
        self.assertEqual(sha(c.POLICY), POLICY_SHA)
        self.assertEqual(sha(c.HERE/'study_plan.json'), PLAN_SHA)
        self.assertEqual(len(c.plan()['required_cases']), 45)
        self.assertEqual(sum(len(s['required_keys']) for s in c.plan()['required_cases']), 2342)
        old_manifest = read(c.PARENT_EVIDENCE/'source_manifest.json')
        for name in ('numerics.py','smoke.py','runner.py','trainer.py','report.py','state.py','preflight.py','process_env.py','pipeline.py'):
            file=c.HERE/name
            self.assertEqual(sha(file), old_manifest['files'][str(file.relative_to(c.ROOT))])

    def test_entry_failure_saved_before_assertion(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'failure.json'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='not-an-empty-mask',PYTHONDONTWRITEBYTECODE='1')
            proc=subprocess.run([sys.executable,'-B','-m','experiments.mamba3_mimo_time.preflight','--evidence',str(p)],
                                cwd=c.ROOT,env=env,capture_output=True,text=True)
            self.assertNotEqual(proc.returncode,0)
            record=read(p)
            self.assertEqual(record['status'],'FAIL')
            self.assertEqual(record['stages'][0]['stage'],'process_entry')
            self.assertFalse(record['stages'][0]['torch_imported'])
            self.assertIsNone(record['last_successful_stage'])
            self.assertIn('CPU mask absent/changed',record['traceback'])


class RetryTests(unittest.TestCase):
    def test_verified_parent_and_tampering(self):
        verify_parent()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); e=root/c.PARENT_EVIDENCE.relative_to(c.ROOT)
            for source in (c.HERE/'evidence/job4356310', c.PARENT_EVIDENCE):
                shutil.copytree(source,root/source.relative_to(c.ROOT))
                for row in read(source/'preservation_manifest.json')['files']:
                    original=root/row['source'];original.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(root/row['destination'],original)
            m=read(e/'preservation_manifest.json')
            verify_parent(root,live=True)
            unknown=root/c.HERE.relative_to(c.ROOT)/'slurm_logs/attempt_999'
            unknown.mkdir()
            with self.assertRaisesRegex(ValueError,'Unknown attempt'):verify_parent(root,live=True)
            unknown.rmdir()
            p=root/m['absent_paths'][0];p.parent.mkdir(parents=True,exist_ok=True)
            create(p,{'status':'RUNNING'})
            with self.assertRaisesRegex(ValueError,'artifact appeared'):
                verify_parent(root,live=True)
            p.unlink()
            with (e/'slurm_logs/preflight/stderr.log').open('a') as f:f.write('changed')
            with self.assertRaisesRegex(ValueError,'Preserved parent evidence changed'):
                verify_parent(root)

    def test_separate_namespace_and_one_shot(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);logs=root/('slurm_logs/attempt_'+c.EXECUTION_ATTEMPT);runs=root/('runs/attempt_'+c.EXECUTION_ATTEMPT)
            create(root/'slurm_logs/pipeline.lock',{'old':True})
            create(root/'slurm_logs/reservation_001.json',{'old':True})
            create(root/'runs/pilot_summary.json',{'old':True})
            names=dict(LOGS=logs,RUNS=runs,LOGIN=logs/'login_verification_001.json',
                       RESERVATION=logs/'reservation_001.json',SUBMISSION=logs/'submission_001.json',
                       PIPELINE=logs/'pipeline_status.json',GATE=runs/'admission_001.json',
                       SMOKE=runs/'smoke_001.json',SUMMARY=runs/'pilot_summary.json')
            with patch.multiple(c,**names):
                c.unused()
                create(c.RESERVATION,parent_binding())
                with self.assertRaises(FileExistsError):c.unused()
            self.assertEqual(read(root/'slurm_logs/pipeline.lock'),{'old':True})

    def test_serialized_attempt_binding_rejects_old_or_changed_owner(self):
        expected=dict(parent_binding(),execution_commit='a'*40,source_hash='source',source_manifest_sha256='manifest')
        login=dict(expected,status='PASS',tracked_clean=True,historical_sources_verified=True,published_commit='a'*40)
        reserve=dict(expected,status='RESERVED',login_sha256='login',max_scientific_fits=3,token='b'*32)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            create(p/'login',login);create(p/'reserve',reserve)
            login,reserve=read(p/'login'),read(p/'reserve')
            validate_ownership(login,reserve,'login',expected,'999')
            for key in ('execution_attempt','retry_of_job','source_hash','source_manifest_sha256','parent_failure_evidence_sha256'):
                bad=dict(reserve,**{key:'wrong'})
                with self.assertRaises(ValueError):validate_ownership(login,bad,'login',expected,'999')

    def test_source_manifest_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);here=root/'experiment';here.mkdir()
            (root/'source.py').write_text('changed\n')
            files={'source.py':'0'*64}
            create(here/'source_manifest.json',dict(files=files,source_hash=digest(files)))
            with patch.object(c,'ROOT',root),patch.object(c,'HERE',here),self.assertRaisesRegex(ValueError,'Source mismatch'):
                verify()


@unittest.skipUnless(TORCH,'PyTorch unavailable; full CPU preflight requires frozen cluster environment')
class FullPreflightTests(unittest.TestCase):
    def check_record(self, record):
        self.assertEqual(record['status'],'PASS')
        self.assertFalse(record['final_cuda_initialized'])
        self.assertEqual(record['last_successful_stage'],'after_runtime_and_imported_source_verification')
        self.assertEqual(record['cuda_forward'],0)
        self.assertEqual(record['optimizer_steps'],0)
        for stage in record['stages']:
            self.assertEqual(stage['cpu_mask'],dict(present=True,value=''))
            self.assertIsNot(stage['cuda_initialized'],True)
            if 'effective_device' in stage:self.assertEqual(stage['effective_device'],'cpu')
            if 'gpu_id' in stage:self.assertEqual(stage['gpu_id'],'')
            if 'parameter_devices' in stage:
                self.assertEqual(stage['parameter_devices'],['cpu'])
                self.assertTrue(all(x=='cpu' for x in stage['buffer_devices']))
        for name in ('after_checked_config','after_reference_config','after_cpu_model_construction'):
            self.assertEqual(sum(s['stage']==name for s in record['stages']),3)

    def test_real_pipeline_cpu_preflight(self):
        with tempfile.TemporaryDirectory() as d,patch.object(c,'LOGS',Path(d)),patch.dict(os.environ,{CUDA_MASK:'GPU-parent-marker'}):
            p=Path(d)/'preflight/evidence.json'
            pipeline.child('preflight','experiments.mamba3_mimo_time.preflight',['--evidence',str(p)],time.time()+600)
            self.check_record(read(p))
            self.assertEqual(os.environ[CUDA_MASK],'GPU-parent-marker')

    def test_real_launcher_without_git(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'evidence.json'
            env=dict(os.environ,PATH='/nonexistent',CUDA_VISIBLE_DEVICES='GPU-parent-marker',REPO_ROOT=str(c.ROOT))
            proc=subprocess.run(['/bin/bash',str(c.ROOT/'slurm/mamba3_mimo_time.sh'),'--preflight-only','--evidence',str(p)],
                                env=env,cwd=c.ROOT,capture_output=True,text=True,timeout=600)
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.check_record(read(p))
