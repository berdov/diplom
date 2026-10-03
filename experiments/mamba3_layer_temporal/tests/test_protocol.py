import copy
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import ExitStack,contextmanager
from pathlib import Path
from unittest.mock import patch
from experiments.mamba3_layer_temporal import config as c,provenance,report,pipeline,submit
from experiments.mamba3_layer_temporal.process_env import child_environment
from experiments.mamba3_mimo_time.records import read,create,sha,Registry,accepted_cases


@contextmanager
def scratch():
    with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
        root=Path(tmp)
        values=dict(ROOT=root,LOGS=root/'logs',RUNS=root/'runs',LAUNCHER=root/'launch.sh')
        for key in ('LOGIN','RESERVATION','SUBMISSION','PIPELINE'):values[key]=root/'logs'/(key+'.json')
        for key in ('INHERITED','GATE','SMOKE','SUMMARY'):values[key]=root/'runs'/(key+'.json')
        for k,v in values.items():stack.enter_context(patch.object(c,k,v))
        yield root


class ProtocolTests(unittest.TestCase):
    def test_plan_scope(self):
        p=c.plan();self.assertEqual(p['max_scientific_fits'],2);self.assertEqual(p['max_jobs'],1)
        self.assertEqual(p['test_evaluations'],0);self.assertEqual(p['automatic_retries'],0)
        self.assertEqual(p['budget']['allocation_seconds'],21600)
        self.assertEqual(p['diagnostic_grid'],[0,.01,.1,.25,.5,1,2,4,10,100])

    def test_registry_missing_failed_duplicate(self):
        specs=c.plan()['required_cases'];rows=[dict(case_id=s['id'],status='PASS',required_keys=s['required_keys'],checks={k:dict(passed=True) for k in s['required_keys']}) for s in specs]
        self.assertTrue(accepted_cases(rows,specs))
        for damage in ('missing','failed','duplicate'):
            bad=copy.deepcopy(rows)
            if damage=='missing':bad[0]['checks'].pop(specs[0]['required_keys'][0])
            elif damage=='failed':bad[0]['checks'][specs[0]['required_keys'][0]]['passed']=False
            else:bad.append(copy.deepcopy(bad[0]))
            self.assertFalse(accepted_cases(bad,specs))

    def test_failure_progress_durable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'gate.json';value={};create(path,value);spec=dict(id='fixture',required_keys=['a','b']);r=Registry(path,value,[spec])
            def failure(save):save(dict(checks={'a':dict(passed=False)},required_keys=['a']));raise ValueError('fixture')
            with self.assertRaises(RuntimeError):r.run(spec,failure)
            self.assertFalse(read(path)['cases'][0]['checks']['a']['passed'])

    def test_one_shot_submission(self):
        with scratch(),patch.object(submit,'bindings',return_value={'execution_commit':'a'*40}):
            def sbatch(*args,**kwargs):
                self.assertTrue(c.RESERVATION.exists());self.assertTrue(c.SUBMISSION.exists())
                return subprocess.CompletedProcess(args[0],0,'1234\n','')
            with patch.object(submit.subprocess,'run',side_effect=sbatch) as call:
                submit.reserve_and_submit(dict(execution_commit='a'*40),dict(source_hash='b'*64))
                with self.assertRaises(FileExistsError):submit.reserve_and_submit({}, {})
                self.assertEqual(call.call_count,1)

    def test_ambiguous_submission_blocks_retry(self):
        with scratch(),patch.object(submit,'bindings',return_value={}):
            with patch.object(submit.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'unknown','')) as call:
                with self.assertRaises(RuntimeError):submit.reserve_and_submit(dict(execution_commit='a'*40),dict(source_hash='b'*64))
                with self.assertRaises(FileExistsError):submit.reserve_and_submit({}, {})
                self.assertEqual(call.call_count,1);self.assertEqual(read(c.SUBMISSION)['status'],'SUBMISSION_UNKNOWN_NO_RETRY')

    def test_child_gpu_visibility(self):
        parent={'CUDA_VISIBLE_DEVICES':'GPU-owned'}
        for stage in ('gate','smoke',*c.MODES):self.assertEqual(child_environment(stage,parent)['CUDA_VISIBLE_DEVICES'],'GPU-owned')
        self.assertEqual(child_environment('preflight',parent)['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(parent,{'CUDA_VISIBLE_DEVICES':'GPU-owned'})
        with self.assertRaises(ValueError):child_environment('TEST',parent)

    def test_immutable_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.json';create(p,{'a':1});digest=sha(p)
            with self.assertRaises(FileExistsError):create(p,{'a':2})
            self.assertEqual(sha(p),digest)

    def test_incomplete_summary_has_no_contrast(self):
        r=report.summarize({});self.assertEqual(r['status'],'INCOMPLETE');self.assertIsNone(r['delta']);self.assertIsNone(r['first27_delta'])

    def test_smoke_requires_learning_and_divergence(self):
        rows=[]
        for v in c.MODES:
            rows.append(dict(temporal_sharing=v,status='PASS',steps=[dict(step=i,loss=1.,finite_loss=True,gradient_norms=dict.fromkeys(c.parameter_keys(v),1.)) for i in range(3)],roundtrip_passed=True,roundtrip='weights_only=True',diverged_parameters=['calibrators.decay.last.weight']))
        r=dict(batch=2048,history_length=50,kernel_length=56,steps_per_mode=3,rows=rows);provenance.validate_smoke(r)
        bad=copy.deepcopy(r);bad['rows'][1]['diverged_parameters']=[]
        with self.assertRaises(ValueError):provenance.validate_smoke(bad)
        bad=copy.deepcopy(r);bad['rows'][1]['steps'][1]['gradient_norms'].pop('layer1_times.calibrators.decay.first.weight')
        with self.assertRaises(ValueError):provenance.validate_smoke(bad)

    def test_replay_rejects_history_or_checkpoint_change(self):
        r=read(c.PILOT);r['initial_common_calibrator_hashes']=r['initial_calibrator_hashes'];report.replay_check(r)
        for field in ('checkpoint_sha256','history'):
            bad=copy.deepcopy(r)
            if field=='history':bad['history'][0]['train_loss']+=.01
            else:bad[field]='wrong'
            with self.assertRaises(ValueError):report.replay_check(bad)


if __name__=='__main__':unittest.main()
