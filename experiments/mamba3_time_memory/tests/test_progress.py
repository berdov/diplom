"""Real first-create/owned-update regression, with no Torch/model dependency.

Runner integration compiles its unchanged production main body and supplies
only expensive external dependencies as doubles. Its filesystem, progress
writer and exception handling are the production implementations.
"""
import argparse
import ast
import copy
import os
import signal
import sys
import tempfile
import traceback
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.mamba3_mimo_time.records import create,read,update,now
from experiments.mamba3_time_memory import progress


def initial_record():
    return dict(execution_attempt='002',study_id='mamba3_time_addressed_memory_pilot_001',
                execution_commit='a'*40,source_hash='b'*64,job_id='900002',
                run_id='mamba3_time_memory_no_memory_seed2026_001',memory_mode='no_memory',seed=2026,
                status='RUNNING',stage='SETUP',scientific_fit_started=False,actual_epochs=0,
                history=[],started_at=now())


@contextmanager
def scratch():
    with tempfile.TemporaryDirectory(prefix='time-memory-progress-regression-') as directory:
        root=Path(directory);runtime=root/'run'
        yield dict(runtime=runtime,run_id='mamba3_time_memory_no_memory_seed2026_001',
                   result=root/'result.json',lock=runtime/'run.lock',
                   checkpoint=runtime/'best_state_dict.pth',metadata=runtime/'best_metadata.json')


def production_main(paths,train):
    source=Path(progress.__file__).with_name('runner.py')
    tree=ast.parse(source.read_text(),filename=str(source))
    main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
    base={key:value for key,value in initial_record().items()
          if key in progress.OWNER_FIELDS and key not in ('run_id','memory_mode','seed')}
    namespace=dict(__name__='progress_runner_fixture',argparse=argparse,os=os,signal=signal,
                   traceback=traceback,identity=lambda:base,require_stage=lambda *_:'c'*64,
                   c=SimpleNamespace(MODES=('no_memory','index_memory','time_memory'),
                                     GATE=Path('gate.json'),SMOKE=Path('smoke.json'),paths=lambda _:paths),
                   torch=SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=None))),
                                         cuda=SimpleNamespace(OutOfMemoryError=MemoryError)),
                   create=create,read=read,update=update,now=now,write_progress=progress.write,train=train)
    exec(compile(ast.Module(body=[main],type_ignores=[]),str(source),'exec'),namespace)
    before=Path.cwd()
    try:
        with patch.object(sys,'argv',['runner','--variant','no_memory']),patch.object(signal,'signal'):
            namespace['main']()
    finally:
        os.chdir(before)


class ProgressTests(unittest.TestCase):
    def test_first_write_creates_setup_before_any_fit_then_updates_epochs(self):
        with scratch() as paths:
            record=initial_record();path=paths['runtime']/'progress.json'
            self.assertFalse(path.exists())
            progress.write(record,paths)
            first=read(path)
            self.assertEqual(first['stage'],'SETUP')
            self.assertIs(first['scientific_fit_started'],False)
            self.assertIsNone(first['last_epoch'])
            record.update(stage='TRAIN_VALID',scientific_fit_started=True,actual_epochs=1,history=[dict(epoch=0)])
            progress.write(record,paths)
            second=read(path)
            self.assertIs(second['scientific_fit_started'],True)
            self.assertEqual(second['last_epoch'],0)
            self.assertEqual(second['actual_epochs'],1)
            self.assertTrue(all(first[key]==second[key] for key in progress.OWNER_FIELDS))
            record.update(status='PASS',stage='COMPLETED',finished_at=now())
            progress.write(record,paths)
            self.assertEqual(read(path)['status'],'PASS')

    def test_foreign_owner_never_changes_existing_progress_bytes(self):
        with scratch() as paths:
            record=initial_record();progress.write(record,paths)
            path=paths['runtime']/'progress.json';before=path.read_bytes()
            for key in progress.OWNER_FIELDS:
                changed=copy.deepcopy(record)
                changed[key]=2027 if key=='seed' else 'different-owner'
                with self.subTest(owner_key=key),self.assertRaises(ValueError):
                    progress.write(changed,paths)
                self.assertEqual(path.read_bytes(),before)

    def test_missing_owner_is_rejected_without_creating_file(self):
        with scratch() as paths:
            record=initial_record();del record['job_id']
            with self.assertRaises(ValueError):progress.write(record,paths)
            self.assertFalse((paths['runtime']/'progress.json').exists())

    def test_symlink_progress_is_not_followed(self):
        with scratch() as paths:
            paths['runtime'].mkdir();target=paths['runtime'].parent/'foreign.json'
            create(target,dict(foreign=True));before=target.read_bytes()
            (paths['runtime']/'progress.json').symlink_to(target)
            with self.assertRaises(ValueError):progress.write(initial_record(),paths)
            self.assertEqual(target.read_bytes(),before)

    def test_production_runner_initializes_progress_before_train_callback(self):
        with scratch() as paths:
            called=[]
            def train(variant,record,actual_paths):
                saved=read(paths['runtime']/'progress.json')
                self.assertEqual(variant,'no_memory');self.assertIs(actual_paths,paths)
                self.assertEqual(saved['stage'],'SETUP');self.assertIs(saved['scientific_fit_started'],False)
                self.assertEqual(record['stage'],'SETUP');self.assertIs(record['scientific_fit_started'],False)
                called.append(True)
                record.update(scientific_fit_started=True,stage='TRAIN_VALID')
                progress.write(record,paths)
                record.update(status='PASS',stage='COMPLETED',history=[dict(epoch=0)],actual_epochs=1)
            production_main(paths,train)
            self.assertEqual(called,[True])
            self.assertEqual(read(paths['runtime']/'progress.json')['stage'],'COMPLETED')
            self.assertEqual(read(paths['result'])['status'],'PASS')

    def test_production_runner_initialization_error_is_saved_before_fit(self):
        with scratch() as paths:
            foreign=initial_record();foreign['job_id']='another-job';progress.write(foreign,paths)
            before=(paths['runtime']/'progress.json').read_bytes();called=[]
            with self.assertRaises(ValueError):production_main(paths,lambda *_:called.append(True))
            self.assertFalse(called)
            raw=read(paths['result'])
            self.assertEqual(raw['status'],'FAIL');self.assertIs(raw['scientific_fit_started'],False)
            self.assertEqual(raw['stage'],'SETUP');self.assertIn('Progress belongs',raw['error'])
            self.assertIn('traceback',raw);self.assertIn('finished_at',raw)
            self.assertEqual((paths['runtime']/'progress.json').read_bytes(),before)


if __name__=='__main__':unittest.main()
