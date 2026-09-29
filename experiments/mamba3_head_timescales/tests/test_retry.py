"""Real temporary files for retry lineage, preservation and stage guards."""
import copy
import os
import shutil
import tempfile
import unittest
from contextlib import contextmanager,ExitStack
from pathlib import Path
from unittest.mock import patch
from experiments.mamba3_head_timescales import config as c,provenance,retry,pipeline
from experiments.mamba3_mimo_time.records import read,create,sha,update


@contextmanager
def parent_fixture():
    with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
        here=Path(tmp)/'study';archive=here/'evidence/job4361071'
        shutil.copytree(c.PRESERVATION.parent,archive)
        manifest=read(archive/'preservation_manifest.json')
        for name,row in manifest['files'].items():
            target=here/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(archive/row['preserved'],target)
        for name,value in dict(HERE=here,PRESERVATION=archive/'preservation_manifest.json',LOGS=here/'slurm_logs/attempt_002',RUNS=here/'runs/attempt_002').items():
            stack.enter_context(patch.object(c,name,value))
        for name,relative in dict(LOGIN='login_verification_001.json',RESERVATION='reservation_001.json',SUBMISSION='submission_001.json',PIPELINE='pipeline_status.json').items():
            stack.enter_context(patch.object(c,name,c.LOGS/relative))
        for name,relative in dict(INHERITED='inherited_kernel_001.json',GATE='targeted_gate_001.json',SMOKE='smoke_001.json',SUMMARY='pilot_summary.json').items():
            stack.enter_context(patch.object(c,name,c.RUNS/relative))
        yield here,archive


class RetryTests(unittest.TestCase):
    def test_exact_parent_and_original_not_run_files_are_preserved(self):
        with parent_fixture() as (here,archive):
            before={p:sha(p) for p in archive.rglob('*') if p.is_file()}
            result=retry.verify_parent(originals=True)
            self.assertEqual(result['status'],'PASS');self.assertEqual(result['parent_scientific_fits'],0)
            self.assertEqual(result['parent_numerical_checks_preserved'],198)
            retry.check_attempt_namespaces()
            self.assertTrue((here/'runs/mamba3_headtime_fixed_seed2026_001.json').exists())
            self.assertEqual(before,{p:sha(p) for p in before})

    def test_original_modified_or_scientific_lock_blocks_retry(self):
        for damage in ('raw','lock','checkpoint','smoke'):
            with self.subTest(damage=damage),parent_fixture() as (here,archive):
                if damage=='raw':
                    p=here/'runs/mamba3_headtime_fixed_seed2026_001.json';r=read(p);r['scientific_fit_started']=True;update(p,r)
                elif damage=='smoke':create(here/'runs/smoke_001.json',dict(status='PASS'))
                else:
                    runtime=here/'slurm_logs/mamba3_headtime_fixed_seed2026_001'
                    create(runtime/('run.lock' if damage=='lock' else 'checkpoints/best_metadata.json'),{})
                with self.assertRaises((ValueError,FileNotFoundError)):retry.verify_parent(originals=True)

    def test_archive_change_and_unknown_attempt_are_rejected(self):
        with parent_fixture() as (here,archive):
            p=archive/'files/runs/pilot_summary.json';p.write_text('{}')
            with self.assertRaises(ValueError):retry.verify_parent()
        with parent_fixture() as (here,archive):
            (here/'runs/attempt_003').mkdir()
            with self.assertRaises(ValueError):retry.check_attempt_namespaces()
        with parent_fixture() as (here,archive):
            create(here/'slurm_logs/stray/reservation_999.json',{})
            with self.assertRaises(ValueError):retry.check_attempt_namespaces()
        with parent_fixture() as (here,archive):
            create(c.RESERVATION,{})
            with self.assertRaises((ValueError,FileExistsError)):retry.check_attempt_namespaces()

    def test_gate_stage_requires_complete_current_registry_from_disk(self):
        with parent_fixture():
            base=dict(execution_attempt='002',job_id='fixture')
            create(c.GATE,dict(base,status='PASS',cases=[]))
            with self.assertRaises(ValueError):provenance.require_stage(c.GATE,base)
            wrong=dict(base,status='PASS',execution_attempt='001',cases=[]);update(c.GATE,wrong)
            with self.assertRaises(ValueError):provenance.require_stage(c.GATE,base)
            with self.assertRaises(FileNotFoundError):provenance.require_stage(c.SMOKE,base)

    def test_namespace_and_scientific_plan_unchanged(self):
        self.assertEqual(c.EXECUTION_ATTEMPT,'002')
        self.assertEqual(c.LOGS,c.HERE/'slurm_logs/attempt_002')
        self.assertEqual(c.RUNS,c.HERE/'runs/attempt_002')
        plan=c.plan()
        self.assertEqual(len(plan['required_cases']),9)
        self.assertEqual(sum(len(x['required_keys']) for x in plan['required_cases']),228)
        self.assertEqual(sha(c.HERE/'study_plan.json'),sha(c.PRESERVATION.parent/'files/study_plan.json'))
        for mode in c.MODES:
            self.assertTrue(c.paths(mode)['runtime'].is_relative_to(c.LOGS))
            self.assertTrue(c.paths(mode)['result'].is_relative_to(c.RUNS))
            values=c.settings(mode,'cpu')
            self.assertEqual(values['three_time_mode'],'dual');self.assertEqual(values['time_scale_mode'],mode)
            self.assertEqual(values['gpu_id'],'');self.assertIs(values['use_gpu'],False)
            self.assertEqual(values['seed'],2026)

    def test_failed_pipeline_with_real_retry_binding_saves_not_run(self):
        with parent_fixture(),ExitStack() as stack:
            base=dict(retry.retry_bindings(),job_id='fixture')
            stack.enter_context(patch.dict(os.environ,{},clear=False))
            for name,value in (('identity',base),('inherited',{}),('runtime',{}),('require_stage','fixture')):
                stack.enter_context(patch.object(pipeline,name,return_value=value))
            stack.enter_context(patch.object(pipeline.signal,'signal'))
            stack.enter_context(patch.object(pipeline,'child',side_effect=RuntimeError('fixture gate failure')))
            self.assertEqual(pipeline.main(),1)
            result=read(c.SUMMARY)
            self.assertEqual(result['status'],'INCOMPLETE')
            self.assertEqual(result['scientific_fits_started'],0)
            for mode in c.MODES:
                row=read(c.paths(mode)['result'])
                self.assertEqual(row['status'],'NOT_RUN')
                self.assertEqual(row['retry_reason'],retry.REASON)
                self.assertIn('fixture gate failure',row['reason'])
                self.assertEqual(row['history'],[])
            self.assertTrue(c.SUMMARY.with_suffix('.md').exists())


if __name__=='__main__':unittest.main()
