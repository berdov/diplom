import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from experiments.mamba3_mimo_time import config as c

TORCH=importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH, 'PyTorch unavailable: CPU regression SKIP, not GPU evidence')
class CPUTests(unittest.TestCase):
    def test_real_adam_json_roundtrip_and_all_mutations(self):
        import torch
        from experiments.mamba3_three_time.confirmation.state import canonical_optimizer_settings, compare_optimizer_settings
        def metadata():
            a=torch.nn.Parameter(torch.ones(2));b=torch.nn.Parameter(torch.zeros(1))
            optimizer=torch.optim.Adam([{'params':[a]},{'params':[b],'lr':.002}],lr=.001)
            return [{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups]
        raw=metadata();canonical=canonical_optimizer_settings(raw)
        self.assertIsInstance(raw[0]['betas'],tuple)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'optimizer.json';p.write_text(json.dumps(canonical))
            stored=json.loads(p.read_text());compare_optimizer_settings(stored,metadata())
        changes={'lr':.5,'betas':[.8,.95],'eps':1e-4,'weight_decay':.1,
                 'amsgrad':True,'maximize':True,'foreach':False,'capturable':True,
                 'differentiable':True,'fused':False,'decoupled_weight_decay':True}
        for key,value in changes.items():
            changed=copy.deepcopy(stored);changed[0][key]=value
            with self.assertRaises(ValueError,msg=key):
                compare_optimizer_settings(stored,changed)
        for key in stored[0]:
            changed=copy.deepcopy(stored);changed[0].pop(key)
            with self.assertRaises(ValueError,msg=key):
                compare_optimizer_settings(stored,changed)
        with self.assertRaises(ValueError):
            compare_optimizer_settings(stored,list(reversed(stored)))
        with self.assertRaises(ValueError):
            compare_optimizer_settings([{'flag':None}],[{'flag':False}])

    def test_actual_counts_effective_flags_and_base_pairing(self):
        import torch
        from experiments.mamba3_mimo_time.state import initialization, paired
        result=initialization('cpu')
        rows=json.loads(json.dumps(result['rows']))
        self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual([r['parameter_count'] for r in rows],[714888,715020,715086])
        for a,b in ((0,1),(0,2),(1,2)):
            paired(rows[a],rows[b])
        rows[2]['initial_calibrator_hashes']['phase']='changed'
        with self.assertRaises(ValueError):
            paired(rows[1],rows[2])
        rows[0]['initial_calibrator_hashes']={'scan':'dummy'}
        with self.assertRaises(ValueError):
            paired(rows[0],rows[1])

    def test_numerical_zero_missing_nonfinite(self):
        import torch
        from experiments.mamba3_mimo_time.numerics import compare,residual
        z=torch.zeros(1,4,2);one=torch.ones_like(z)
        self.assertTrue(compare(z,z,'vjp')['passed'])
        self.assertFalse(compare(one,z,'vjp')['passed'])
        self.assertFalse(compare(None,z)['passed'])
        self.assertEqual(residual(z,2)['zero_reference_status'],'ZERO_SIGNAL')
        self.assertFalse(residual(z,2)['passed'])
        z[:,2:]=1
        self.assertEqual(residual(z,2)['rho'],1)
        self.assertFalse(residual(z,2)['passed'])
        z.fill_(float('nan'))
        self.assertFalse(residual(z,2)['passed'])
        json.dumps(residual(z,2),allow_nan=False)

    def test_fit_algorithm_and_safe_checkpoint_callback_inherited(self):
        from recbole.trainer import Trainer
        from experiments.mamba3_mimo_time.trainer import trainer_class
        cls=trainer_class(Trainer,object(),{}, {})
        self.assertIs(cls.fit,Trainer.fit)
        self.assertEqual(cls._save_checkpoint.__module__,'experiments.mamba3_context_time.trainer')
        self.assertEqual(cls.evaluate.__module__,'experiments.mamba3_context_time.trainer')
        self.assertEqual(cls._valid_epoch.__module__,'experiments.mamba3_mimo_time.trainer')
