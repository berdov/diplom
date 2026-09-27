"""Real runtime verifiers with Git forbidden; fixtures never claim model PASS."""
import contextlib
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.mamba3_three_time.confirmation import config, provenance as p

HAS_RECBOLE = importlib.util.find_spec('recbole') is not None
FIXTURE_COMMIT = 'f' * 40


def fixture(directory, m):
    login = dict(**p.bindings(FIXTURE_COMMIT, m), status='PASS', tracked_sources_clean=True,
                 historical_git_objects_verified=True, published_branch_commit=FIXTURE_COMMIT,
                 test_fixture=True)
    p.create_record(directory / 'login.json', login)
    submission = dict(**p.bindings(FIXTURE_COMMIT, m), status='RESERVED', retry_of_job='4354908',
        retry_reason='launcher_git_unavailable_before_python', parent_execution_commit=config.PARENT_COMMIT,
        old_source_hash=config.PARENT_HASH, new_execution_commit=FIXTURE_COMMIT, new_source_hash=m['source_hash'],
        scientific_runs_started_in_parent=0, outputs_and_locks_absent=True,
        parent_failure_sha256=p.sha(config.FAILED / 'failure.json'),
        login_verification_sha256=p.sha(directory / 'login.json'), test_fixture=True)
    p.create_record(directory / 'submission.json', submission)
    p.create_record(directory / 'test_fixture.json', dict(runtime_only=True, scientific_fits=0))
    return dict(RUN_COMMIT=FIXTURE_COMMIT, EXPECTED_STUDY_HASH=m['source_hash'], EXPECTED_CORE_HASH=p.CORE,
                SLURM_JOB_ID='12345', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(config.ROOT))


@contextlib.contextmanager
def no_git():
    original = subprocess.Popen
    attempted = []
    def guarded(command, *args, **kwargs):
        tokens = command if isinstance(command, (list, tuple)) else str(command).split()
        if any(Path(str(token)).name in ('git', 'git.exe') for token in tokens):
            attempted.append(command)
            raise AssertionError('Git is unavailable in runtime regression')
        return original(command, *args, **kwargs)
    with patch('subprocess.Popen', side_effect=guarded):
        yield attempted


class StaticContracts(unittest.TestCase):
    def test_historical_manifest_matches_git_objects(self):
        self.assertEqual(p.historical_from_git(), json.loads((config.HERE / 'historical_sources.json').read_text()))

    def test_launcher_has_no_git_and_explicit_python(self):
        text = (config.ROOT / 'slurm/mamba3_three_time_confirmation.sh').read_text()
        self.assertNotIn('git ', text)
        self.assertIn('--runtime-preflight-only', text)
        self.assertIn('/home/daryumin/iberdov/diplom/envs/mamba3/bin/python', text)
        self.assertNotIn('export PATH=', text)

    def test_duplicate_reservation_and_scientific_lock_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reservation = root / 'submission_002.json'
            p.create_record(reservation, {'status': 'RESERVED'})
            with patch.object(config, 'SUBMISSION', reservation):
                with self.assertRaises(FileExistsError): config.unused(include_submission=True)
            with patch.multiple(config, LOGS=root/'logs', RUNS=root/'runs'):
                lock = config.paths('dual', 2027)['lock']
                p.create_record(lock, {'scientific_fit_started': True})
                with self.assertRaises(FileExistsError): config.unused()
            with patch.object(config, 'RUNS', root/'results'):
                result = config.paths('dual', 2027)['result']
                p.create_record(result, {'status': 'RUNNING'})
                with self.assertRaises(FileExistsError): config.unused()

    def test_historical_manifest_tampering_rejected_without_git(self):
        with tempfile.TemporaryDirectory() as folder:
            here = Path(folder)
            value = json.loads((config.HERE / 'historical_sources.json').read_text())
            value['files'][next(iter(value['files']))]['sha256'] = '0'*64
            p.create_record(here/'historical_sources.json', value)
            with patch.object(p, 'HERE', here), no_git() as attempted:
                with self.assertRaisesRegex(ValueError, 'Historical execution source mismatch'):
                    p.execution_sources()
                self.assertFalse(attempted)


@unittest.skipUnless(HAS_RECBOLE, 'Full runtime regression requires existing cluster RecBole/Mamba environment')
class RuntimeContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.m = p.verify()
        self.env = fixture(self.directory, self.m)

    def run_identity(self):
        return p.identity(submission_path=self.directory/'submission.json', login_path=self.directory/'login.json')

    def test_real_runtime_without_git_and_fast_start(self):
        with patch.dict(os.environ, self.env), no_git() as attempted:
            row = self.run_identity()
            self.assertEqual(row['execution_commit'], FIXTURE_COMMIT)
            self.assertEqual(row['submission_attempt'], 2)
            self.assertIn('not runtime Git HEAD', row['execution_commit_verification'])
            self.assertFalse(attempted)
            submission = json.loads((self.directory/'submission.json').read_text())
            submission.update(status='SUBMITTED', job_id='12345')
            p.atomic_json(self.directory/'submission.json', submission)
            self.run_identity()
            submission['job_id'] = '99999'
            p.atomic_json(self.directory/'submission.json', submission)
            with self.assertRaises(ValueError): self.run_identity()

    def test_wrong_expected_hash(self):
        with patch.dict(os.environ, dict(self.env, EXPECTED_STUDY_HASH='0'*64)), no_git():
            with self.assertRaisesRegex(ValueError, 'Submission/source mismatch'): self.run_identity()

    def test_missing_and_changed_submission_evidence(self):
        with patch.dict(os.environ, self.env), no_git():
            with self.assertRaises(FileNotFoundError):
                p.identity(submission_path=self.directory/'missing.json', login_path=self.directory/'login.json')
            row = json.loads((self.directory/'submission.json').read_text())
            row['historical_manifest_sha256'] = '0'*64
            p.atomic_json(self.directory/'submission.json', row)
            with self.assertRaisesRegex(ValueError, 'bindings differ'): self.run_identity()

    def test_changed_login_evidence(self):
        row = json.loads((self.directory/'login.json').read_text())
        row['extra'] = 'tampered immutable verification'
        p.atomic_json(self.directory/'login.json', row)
        with patch.dict(os.environ, self.env), no_git():
            with self.assertRaisesRegex(ValueError, 'Immutable login evidence changed'): self.run_identity()

    def test_changed_actual_source_rejected(self):
        root = self.directory/'source_fixture'
        for name in self.m['files']:
            dest = root/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(config.ROOT/name, dest)
        here = root/config.HERE.relative_to(config.ROOT)
        shutil.copyfile(config.HERE/'source_manifest.json', here/'source_manifest.json')
        changed = here/'preflight.py'
        changed.write_bytes(changed.read_bytes() + b'\n# changed fixture\n')
        with patch.multiple(p, ROOT=root, HERE=here, PILOT=here.parent/'validation_pilot', FAILED=here/'evidence/submission_001'), no_git():
            with self.assertRaisesRegex(ValueError, 'Confirmation sources changed'): p.verify()

    @unittest.skipUnless(str(config.ROOT)=='/home/daryumin/iberdov/diplom', 'Launcher fixture uses canonical cluster interpreter')
    def test_runtime_shell_no_git(self):
        import torch
        # Empty PATH makes Git impossible, without replacing it with a fake binary.
        env = dict(os.environ, **self.env)
        env['PATH'] = ''
        result = subprocess.run(['/bin/bash',str(config.ROOT/'slurm/mamba3_three_time_confirmation.sh'),
            '--runtime-preflight-only',str(self.directory)],env=env,cwd=config.ROOT,text=True,capture_output=True,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        value = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(value['status'],'PASS')
        self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual(set(p.name for p in self.directory.iterdir()), {'login.json','submission.json','test_fixture.json'})


if __name__=='__main__':
    unittest.main()
