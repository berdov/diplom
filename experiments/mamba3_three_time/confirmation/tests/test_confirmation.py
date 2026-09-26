import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from experiments.mamba3_three_time.confirmation import config, state, report, pipeline, runner, submit, provenance
from experiments.mamba3_three_time.evidence import create_record


def fixture(seed, mode, score=.1, epochs=27):
    h=[dict(epoch=i,valid_ndcg10=score,valid_metrics={'ndcg@10':score},diagnostics={}) for i in range(epochs)]
    return dict(seed=seed,mode=mode,status='PASS',TEST='NOT_RUN',test_evaluation_count=0,history=h,
                actual_epochs=epochs,parameter_count=config.COUNTS[mode],best_epoch=epochs-1,
                best_valid_score=score,best_valid_metrics={'ndcg@10':score},best_diagnostics={},first27_best_ndcg10=score)


class Contracts(unittest.TestCase):
    def test_tasks_ids_and_seed_allowlist(self):
        tasks=config.plan()['tasks']
        self.assertEqual(len(tasks),8)
        self.assertEqual(len({r['run_id'] for r in tasks}),8)
        self.assertEqual([(r['seed'],r['mode']) for r in tasks],[(s,m) for s in config.SEEDS for m in config.MODES])
        for seed in (2026,2031,2027.0):
            with self.assertRaises(ValueError): config.task('dual',seed)
        for mode in ('base','MIMO','stable_adt'):
            with self.assertRaises(ValueError): config.task(mode,2027)

    def test_raw_config_only_allowed_diffs(self):
        for seed in config.SEEDS:
            for mode in config.MODES:
                a=config.pilot(mode)['config']; b=config.settings(mode,seed)
                self.assertEqual({k for k in set(a)|set(b) if a.get(k)!=b.get(k)}, {'seed','checkpoint_dir'})
                self.assertEqual(b['metrics'],['Hit','NDCG','Recall'])

    @unittest.skipUnless(importlib.util.find_spec('recbole'), 'RecBole unavailable locally; mandatory on cluster')
    def test_effective_configuration(self):
        from recbole.config import Config
        from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
        for seed in config.SEEDS:
            for mode in config.MODES:
                values=config.settings(mode,seed)
                values.update(use_gpu=False,device='cpu')
                c=Config(model=ThreeTimeMamba3Rec,config_dict=values)
                config.effective_check(c,mode,seed,cpu=True)
                c['learning_rate']=.002
                with self.assertRaises(ValueError): config.effective_check(c,mode,seed,cpu=True)
        self.assertFalse(torch.cuda.is_initialized())

    def test_frozen_files_and_execution_sources(self):
        old=json.loads((config.PILOT/'source_manifest.json').read_text())
        self.assertEqual(old['source_hash'],config.PILOT_HASH)
        for p,h in old['files'].items():
            self.assertEqual(hashlib.sha256((config.ROOT/p).read_bytes()).hexdigest(),h,p)
        self.assertEqual(hashlib.sha256((config.PILOT/'numeric_acceptance_v1.json').read_bytes()).hexdigest(),config.POLICY_SHA)
        self.assertGreater(len(provenance.execution_sources()),100)

    def test_strict_mapping_all_keys_and_first_linear(self):
        class Net(torch.nn.Module):
            def __init__(self,prefix):
                super().__init__()
                self.backbone=torch.nn.Linear(2,2)
                module=torch.nn.Module()
                module.calibrators=torch.nn.ModuleDict({n:torch.nn.Sequential(torch.nn.Linear(1,3),torch.nn.Linear(3,2)) for n in ('decay','scan')})
                self.add_module(prefix,module)
        a,b=Net('mechanisms'),Net('times')
        state.strict_transfer(a,b)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in state.normalized(a.state_dict()).items()))
        b.extra=torch.nn.Parameter(torch.ones(1))
        with self.assertRaises(ValueError): state.strict_transfer(a,b)
        del b.extra
        b.double()
        with self.assertRaises(ValueError): state.strict_transfer(a,b)

    def test_rng_observation_restore_and_loader(self):
        class Loader: generator=torch.Generator().manual_seed(2027)
        before=state.capture_rng()
        a=state.rng_record(Loader)
        self.assertEqual(a,state.rng_record(Loader))
        torch.rand(10)
        state.restore_rng(before)
        self.assertEqual(a,state.rng_record(Loader))
        self.assertEqual(set(a),{'python','numpy','cpu','cuda','aggregate','loader_generator'})

    def test_numerical_failures_and_bitwise(self):
        a=torch.tensor([1.,2.])
        r=state.difference(a,a.clone())
        self.assertTrue(r['passed'] and r['bitwise_equal'])
        self.assertEqual(r['l2_error'],0)
        self.assertFalse(state.difference(a,a+.1)['passed'])
        self.assertFalse(state.difference(torch.tensor([0.]),torch.tensor([-0.]))['bitwise_equal'])
        self.assertFalse(state.difference(a,torch.tensor([float('nan'),1.]))['passed'])
        r=state.compare_named({'x':a},{'y':a})
        self.assertEqual(r['failed_names'],['x','y'])

    def test_pair_components_and_calibrators(self):
        a={k:'same' for k in ('initial_backbone_sha256','rng_components','rng_before_fit_sha256','protocol','manifest_sha256','train_time_stats_sha256','verified_history_stats','first_train_batch_sha256','precision','optimizer_settings')}
        b=copy.deepcopy(a)
        a['initial_calibrator_hashes']={'decay':'d','scan':'s'}
        b['initial_calibrator_hashes']={'decay':'d','write':'s','phase':'s'}
        state.paired(a,b,True)
        for key in ('rng_components','first_train_batch_sha256'):
            bad=copy.deepcopy(b);bad[key]='different'
            with self.assertRaises(ValueError): state.paired(a,bad,True)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'old.json'
            create_record(p,{'old':True})
            with self.assertRaises(FileExistsError): create_record(p,{'old':False})
            with patch.object(config,'BATCH',p):
                with self.assertRaises(FileExistsError): config.unused()
            self.assertEqual(json.loads(p.read_text()),{'old':True})

    def test_aggregate_signs_ddof_and_incomplete(self):
        records={(s,m):fixture(s,m,.1+(.01 if m=='triple' else 0)) for s in (2026,*config.SEEDS) for m in config.MODES}
        records[(2028,'triple')]=fixture(2028,'triple',.09)
        records[(2029,'triple')]=fixture(2029,'triple',.10)
        r=report.build(records)['new_four_pairs']['full']
        self.assertEqual((r['positive'],r['negative'],r['zero']),(2,1,1))
        self.assertEqual(r['n_available'],4)
        self.assertAlmostEqual(r['paired_delta']['mean'],.0025)
        self.assertIsNotNone(r['paired_delta']['sample_std_ddof1'])
        records[(2030,'triple')].update(status='NOT_RUN',error='budget')
        r=report.build(records)['new_four_pairs']['full']
        self.assertTrue(r['incomplete'])
        self.assertEqual(r['n_available_runs'],7)
        self.assertEqual(r['n_available'],3)
        self.assertEqual(r['rows'][-1]['triple']['reason'],'budget')
        self.assertIsNone(r['relative_gain_percent'])

    def test_first27_short_window_not_equal(self):
        records={(s,m):fixture(s,m,epochs=26 if (s,m)==(2027,'dual') else 27) for s in (2026,*config.SEEDS) for m in config.MODES}
        r=report.build(records)['new_four_pairs']
        self.assertFalse(r['full']['incomplete'])
        self.assertTrue(r['first27']['incomplete'])
        self.assertIsNone(r['first27']['rows'][0]['dual']['ndcg10'])
        records[(2027,'dual')]['test_evaluation_count']=1
        with self.assertRaises(ValueError): report.build(records)

    def test_history_ties_and_failure(self):
        row=fixture(2027,'dual')
        report.validate(row,'dual',2027)
        row['best_epoch']=0
        with self.assertRaises(ValueError): report.validate(row,'dual',2027)
        row=fixture(2027,'dual'); row['history'].pop()
        with self.assertRaises(ValueError): report.validate(row,'dual',2027)

    def test_no_unsafe_load_or_diagnostic_fit_or_test_loader(self):
        from experiments.mamba3_three_time.confirmation import one_batch, initialization
        src=inspect.getsource(runner)
        self.assertIn('del reserved',src)
        self.assertEqual(src.count('FullSortEvalDataLoader(config,'),1)
        self.assertNotIn('next(iter(',src)
        diag=inspect.getsource(one_batch)
        self.assertEqual(diag.count('next(iter(loader))'),1)
        self.assertNotIn('FullSortEvalDataLoader',diag)
        self.assertNotIn('.fit(',diag)
        self.assertNotIn('.evaluate(',diag)
        self.assertNotIn('.fit(',inspect.getsource(initialization))
        for p in config.HERE.glob('*.py'):
            self.assertNotIn('weights_only=False',p.read_text())
            self.assertNotIn('pickle.load',p.read_text())
        self.assertEqual(inspect.getsource(submit).count("['sbatch','--parsable'"),1)
        literals=[n.value for n in ast.walk(ast.parse(inspect.getsource(pipeline))) if isinstance(n,ast.Constant)]
        self.assertNotIn('sbatch',literals)

    def test_gate_failure_stops_before_scientific_fits(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); stages=[]
            def child(module,args,path,deadline):
                stages.append(module.rsplit('.',1)[-1])
                if stages[-1]=='one_batch': raise RuntimeError('injected structural drift')
            def p(mode,seed): return dict(result=root/f'{mode}_{seed}.json')
            base=dict(job_id='fixture',TEST='NOT_RUN',test_evaluation_count=0)
            with patch.multiple(pipeline,LOGS=root,LOCK=root/'lock',BATCH=root/'batch.json',INIT=root/'init.json',SUMMARY=root/'summary.json'), \
                 patch.object(pipeline,'identity',return_value=base),patch.object(pipeline,'reservation'),patch.object(pipeline,'unused'), \
                 patch.object(pipeline,'run_child',side_effect=child),patch.object(pipeline,'paths',side_effect=p), \
                 patch.object(pipeline.signal,'signal'),patch.object(pipeline.traceback,'print_exc'),patch.dict('os.environ'):
                with self.assertRaises(SystemExit): pipeline.main()
            self.assertEqual(stages,['preflight','one_batch','report'])
            self.assertEqual(json.loads((root/'pipeline_status.json').read_text())['scientific_fits'],0)
            for t in config.plan()['tasks']:
                self.assertEqual(json.loads(p(t['mode'],t['seed'])['result'].read_text())['status'],'NOT_RUN')

    def test_missing_checks_block_gate(self):
        r=dict(status='PASS',checks_passed=True,optimizer_steps=2,forward_backward_calls=2,
               structural_tolerance=dict(atol=1e-6,rtol=1e-5),checks={})
        with patch.object(provenance,'require_evidence',return_value=r):
            with self.assertRaises(ValueError): provenance.gates({})

    def test_internal_timeout_keeps_partial_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def child(module,args,path,deadline):
                if module.endswith('.preflight'): raise TimeoutError('deadline fixture')
            def p(mode,seed): return dict(result=root/f'{mode}_{seed}.json')
            base=dict(job_id='fixture',TEST='NOT_RUN',test_evaluation_count=0)
            with patch.multiple(pipeline,LOGS=root,LOCK=root/'lock',BATCH=root/'batch.json',INIT=root/'init.json',SUMMARY=root/'summary.json'), \
                 patch.object(pipeline,'identity',return_value=base),patch.object(pipeline,'reservation'),patch.object(pipeline,'unused'), \
                 patch.object(pipeline,'run_child',side_effect=child),patch.object(pipeline,'paths',side_effect=p), \
                 patch.object(pipeline.signal,'signal'),patch.object(pipeline.traceback,'print_exc'),patch.dict('os.environ'):
                with self.assertRaises(SystemExit): pipeline.main()
            self.assertEqual(json.loads((root/'pipeline_status.json').read_text())['status'],'INCOMPLETE')

    def test_svg_failure_preserves_numeric_summary(self):
        records={(s,m):fixture(s,m) for s in (2026,*config.SEEDS) for m in config.MODES}
        value={'summaries':report.build(records)}
        self.assertIn('paired triple - dual',report.svg(value))
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'summary.json'
            create_record(p,value)
            before=p.read_bytes()
            with self.assertRaises(FileNotFoundError): report.atomic_text(Path(directory)/'missing/summary.svg',report.svg(value))
            self.assertEqual(p.read_bytes(),before)


if __name__=='__main__':
    unittest.main()
