import unittest
import torch
from experiments.mamba3_gap_trap.checks import modulation_checks
from experiments.mamba3_gap_trap.modulation import GapTrap
from experiments.mamba3_gap_trap import config as c
from experiments.mamba3_gap_trap.state import effective_check,initial,paired,transfer_common


class ModulationTests(unittest.TestCase):
    def test_all_modulation_checks(self):
        rows={};modulation_checks('cpu',lambda k,v:rows.update({k:v}))
        self.assertEqual(set(rows),set(c.plan()['required_cases'][2]['required_keys']))
        for k,v in rows.items():
            with self.subTest(check=k):self.assertTrue(v['passed'],v)

    def test_interior_gradcheck(self):
        g=torch.tensor([[1.,838393.,2e6]],dtype=torch.float64,requires_grad=True)
        a=torch.tensor(.4,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x,y:y.clamp(0,1)*x/(x+838393),(g,a)))

    def test_boundary_one_sided(self):
        module=GapTrap().double();g=torch.tensor([[838393.]],dtype=torch.float64);mask=torch.ones_like(g,dtype=torch.bool)
        for value,direction in [(0.,1.),(1.,-1.)]:
            with torch.no_grad():module.alpha.fill_(value)
            y=module(g,mask);derivative=torch.autograd.grad(y.sum(),module.alpha)[0]
            with torch.no_grad():module.alpha.add_(direction*1e-6)
            finite=(module(g,mask)-y.detach())/(direction*1e-6)
            self.assertAlmostEqual(float(derivative),float(finite),places=8)

    def test_invalid(self):
        module=GapTrap()
        for g in (torch.tensor([[-1.]]),torch.tensor([[float('nan')]]),torch.tensor([[float('inf')]])):
            with self.assertRaises(ValueError):module(g,torch.ones_like(g,dtype=torch.bool))
        with self.assertRaises(ValueError):module(torch.zeros(1,1),torch.zeros(1,1))

    def test_counts_state_rng_config(self):
        from recbole.config import Config
        from recbole.utils import init_seed
        from experiments.mamba3_three_time.config import SyntheticCatalog
        from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
        from experiments.mamba3_gap_trap.model import GapTrapMamba3Rec
        records=[];models=[]
        for mode in c.MODES:
            cfg=Config(model=ThreeTimeMamba3Rec,config_dict=c.settings(mode,'cpu'))
            effective_check(cfg,mode)
            init_seed(2026,True)
            net=GapTrapMamba3Rec(cfg,SyntheticCatalog());models.append(net)
            self.assertEqual(sum(p.numel() for p in net.parameters()),c.COUNTS[mode])
            self.assertEqual(set(dict(net.named_parameters())),set(c.plan()['common_parameter_keys'])|({'gap_trap.alpha'} if mode=='gap_trap' else set()))
            records.append(initial(net))
        paired(*records);transfer_common(*models)
        original={k:v.clone() for k,v in models[1].state_dict().items()}
        models[1].load_state_dict(original,strict=True)
        self.assertTrue(all(torch.equal(v,models[1].state_dict()[k]) for k,v in original.items()))
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
