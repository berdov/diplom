import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from experiments.mamba3_gap_trap.centered.confirmation import config as c, report, state, pipeline, submit, continuation
from experiments.mamba3_gap_trap.centered import config as pilot
from experiments.mamba3_gap_trap.centered.process_env import child_environment
from experiments.mamba3_mimo_time.records import read, create, sha


def replay_fixture(seed=2027,variant='fixed_replay'):
    r=copy.deepcopy(read(c.historical(seed)));r.update(run_id=c.paths(variant,seed)['run_id'],gap_trap_mode=variant,
        initial_common_calibrator_hashes=r.pop('initial_calibrator_hashes'),initial_alpha={} if variant=='fixed_replay' else {'gap_trap.alpha':0.},
        parameter_count=c.COUNTS[variant])
    d=read(c.pilot_path('fixed_replay'))['best_diagnostics']['gap_trap']
    for key in ('config','effective_config'):r[key]['gap_trap_mode']=variant
    for row in r['history']:row['diagnostics']['gap_trap']=copy.deepcopy(d)
    r['best_diagnostics']=copy.deepcopy(r['history'][r['best_epoch']]['diagnostics'])
    return r


class ConfirmationTests(unittest.TestCase):
    def test_plan_and_index(self):
        self.assertEqual(c.plan(),c.expected_plan());self.assertEqual(len(c.index()['entries']),8)
        self.assertEqual([(r['seed'],r['variant']) for r in c.tasks()],[(s,v) for s in c.SEEDS for v in c.MODES])

    def test_fresh_paths_and_scope(self):
        paths=[str(c.paths(t['variant'],t['seed'])['result']) for t in c.tasks()]
        self.assertEqual(len(set(paths)),8)
        for seed in (2026,2031):
            with self.assertRaises(ValueError):c.paths('fixed_replay',seed)
        with self.assertRaises(ValueError):c.allocation('003')
        self.assertFalse(set(paths)&{str(pilot.paths(v)['result']) for v in c.MODES})

    def test_exact_settings_except_seed_namespace(self):
        for task in c.tasks():
            actual=c.settings(task['variant'],task['seed']);old=read(c.pilot_path(task['variant']))['config']
            self.assertEqual({k:v for k,v in actual.items() if k not in ('seed','checkpoint_dir')},
                             {k:v for k,v in old.items() if k not in ('seed','checkpoint_dir')})
            self.assertEqual(actual['seed'],task['seed'])

    def test_cpu_initialization_all_new_seeds(self):
        import torch
        from recbole.config import Config
        from recbole.utils import init_seed
        from experiments.mamba3_three_time.config import SyntheticCatalog
        from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
        from experiments.mamba3_gap_trap.centered.model import GapTrapMamba3Rec
        from experiments.mamba3_three_time.confirmation.state import canonical_optimizer_settings
        for seed in c.SEEDS:
            rows=[];batches=[];ref=read(c.historical(seed))
            for variant in c.MODES:
                cfg=Config(model=ThreeTimeMamba3Rec,config_dict=c.settings(variant,seed,device='cpu'))
                state.effective_check(cfg,variant,seed);init_seed(seed,True)
                generator=torch.Generator().manual_seed(seed)
                loader=torch.utils.data.DataLoader(torch.arange(100),batch_size=16,shuffle=True,generator=generator)
                model=GapTrapMamba3Rec(cfg,SyntheticCatalog()).cpu()
                self.assertEqual(sum(p.numel() for p in model.parameters()),c.COUNTS[variant])
                row=state.initial(model,loader);rows.append(row)
                self.assertEqual(row['initial_backbone_sha256'],ref['initial_backbone_sha256'])
                self.assertEqual(row['initial_common_calibrator_hashes'],ref['initial_calibrator_hashes'])
                for key in ('python','numpy','cpu'):self.assertEqual(row['rng_components'][key],ref['rng_components'][key])
                self.assertEqual(row['initial_alpha'],{} if variant=='fixed_replay' else {'gap_trap.alpha':0.})
                opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
                metadata=canonical_optimizer_settings([{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups])
                self.assertEqual(metadata,ref['optimizer_settings'])
                batches.append(next(iter(loader)))
            state.paired(*rows);self.assertTrue(torch.equal(*batches))
        self.assertFalse(torch.cuda.is_initialized())

    def test_historical_schema_mapping(self):
        for seed in c.SEEDS:
            r=replay_fixture(seed);self.assertEqual(report.replay_check(r)['status'],'PASS')
            report.validate_record(r,'fixed_replay',seed)

    def test_replay_rejects_loss_and_checkpoint_drift(self):
        for key in ('loss','checkpoint','rng','calibrator','config'):
            r=replay_fixture()
            if key=='loss':r['history'][0]['train_loss']+=1e-5
            elif key=='checkpoint':r['checkpoint_sha256']='0'*64
            elif key=='rng':r['rng_components']['python']='0'*64
            elif key=='calibrator':r['initial_common_calibrator_hashes']['decay']='0'*64
            else:r['config']['learning_rate']=.002
            with self.assertRaises(ValueError):report.replay_check(r)

    def test_wrong_seed_and_first27_rejected(self):
        r=replay_fixture()
        with self.assertRaises(ValueError):report.validate_record(r,'fixed_replay',2028)
        r['first27_best_ndcg10']=0.
        with self.assertRaises(ValueError):report.validate_record(r,'fixed_replay',2027)

    def test_pair_requires_fields(self):
        a,b=replay_fixture(),replay_fixture(variant='centered_gap_trap')
        state.paired_records(a,b,first_batch=True)
        del a['protocol'];del b['protocol']
        with self.assertRaises(ValueError):state.paired_records(a,b)

    def test_sample_std_signs_and_subset(self):
        lookup={}
        for s,x,y in [(2027,.1,.12),(2028,.2,.19),(2029,.3,.3)]:
            for v,n in zip(c.MODES,(x,y)):lookup[s,v]=dict(status='PASS',ndcg10=n)
        value=report.cohort(lookup,c.SEEDS,'ndcg10')
        self.assertEqual(value['fixed']['n'],3);self.assertAlmostEqual(value['fixed']['sample_std'],.1)
        self.assertEqual(value['signs'],dict(positive=1,negative=1,zero=1))
        self.assertAlmostEqual(value['paired_delta']['mean'],.01/3)

    def test_partial_has_no_confirmatory_aggregate(self):
        r=replay_fixture();value=report.summarize({r['run_id']:r})
        self.assertEqual(value['scientific_fits_completed'],1)
        self.assertEqual(value['status'],'INCOMPLETE');self.assertIsNone(value['primary_new_seeds'])
        self.assertIsNone(value['all5_including_exploratory_pilot'])

    def test_partial_missing_diagnostics_and_nonfinite(self):
        import json
        for status in ('PASS','FAIL'):
            r=replay_fixture(variant='centered_gap_trap');r['status']=status
            del r['history'][0]['diagnostics']['gap_trap'];r['history'][0]['train_loss']=float('nan')
            value=report.summarize({r['run_id']:r})
            self.assertEqual(value['status'],'INCOMPLETE');self.assertEqual(value['scientific_fits_started'],1)
            self.assertIsNone(value['rows'][1]['ndcg10']);json.dumps(value,allow_nan=False)

    def test_centered_saved_seed_drift(self):
        r=replay_fixture(variant='centered_gap_trap');r['config']['seed']=2028
        with self.assertRaises(ValueError):report.validate_record(r,'centered_gap_trap',2027)

    def test_deadline_boundary(self):
        self.assertTrue(pipeline.may_start(6000.,600.))
        self.assertFalse(pipeline.may_start(6000.,600.001))

    def test_child_keeps_visible_gpu(self):
        for v in c.MODES:
            result=child_environment(v,{'CUDA_VISIBLE_DEVICES':'3','PYTHONPATH':'x'})
            self.assertEqual(result['CUDA_VISIBLE_DEVICES'],'3')

    def test_process_directory_is_not_a_scientific_start(self):
        with tempfile.TemporaryDirectory() as folder:
            def paths(v,s):
                root=Path(folder)/f'{v}_{s}'
                return dict(run_id=f'{v}_{s}',runtime=root,result=root/'result.json',lock=root/'lock',checkpoint=root/'checkpoint',metadata=root/'metadata')
            (paths('fixed_replay',2027)['runtime']/'process').mkdir(parents=True)
            with patch.object(c,'paths',paths):self.assertEqual(report.completed_prefix(),[])

    def test_failed_parent_cannot_continue(self):
        with patch.object(continuation,'read',return_value={'status':'FAIL','job_id':'1'}):
            with self.assertRaises((ValueError,KeyError)):continuation.validate(dict(execution_commit='a'*40,source_hash='b'*64,
                scheduler=dict(job_id='1',state='FAILED',exit_code='1:0')),'a'*40,'b'*64)

    def test_one_durable_submission_and_no_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            a={k:Path(folder)/k for k in ('login','reservation','submission','pipeline','lock','inherited','summary','terminal')};a['logs']=Path(folder)
            login=dict(execution_commit='a'*40,status='PASS');manifest=dict(source_hash='b'*64)
            def sbatch(*args,**kwargs):
                self.assertEqual(read(a['submission'])['status'],'SUBMISSION_UNKNOWN_NO_RETRY')
                self.assertEqual(read(a['reservation'])['max_scientific_fits'],8)
                self.assertEqual(read(a['reservation'])['login_sha256'],sha(a['login']))
                return SimpleNamespace(returncode=0,stdout='123456\n',stderr='')
            with patch.object(c,'allocation',return_value=a),patch.object(submit,'bindings',return_value={}),patch.object(submit.subprocess,'run',side_effect=sbatch) as mocked:
                self.assertEqual(submit.reserve_and_submit(login,manifest,'001')['job_id'],'123456')
                with self.assertRaises(FileExistsError):submit.reserve_and_submit(login,manifest,'001')
                self.assertEqual(mocked.call_count,1)
