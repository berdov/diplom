import copy
import io
import unittest
from unittest.mock import patch
import torch
from experiments.mamba3_layer_temporal import config as c
from experiments.mamba3_layer_temporal import model as implementation
from experiments.mamba3_layer_temporal.checks import fresh,cpu_mixer,routing,temporal_aliases,temporal_chain_cpu,nonzero
from experiments.mamba3_layer_temporal.state import initial,paired,effective_check
from experiments.mamba3_layer_temporal.diagnostics import diagnostics
from experiments.mamba3_three_time.calibrators import ThreeTimes,split_dt
from experiments.mamba3_three_time.confirmation.state import rng_record,capture_rng,restore_rng
from experiments.mamba3_timeaware.time_inputs import history_gaps


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared=fresh('shared_layers');cls.shared_rng=rng_record()
        cls.specific=fresh('layer_specific');cls.specific_rng=rng_record()
        cls.historical=fresh('shared_layers',historical=True);cls.historical_rng=rng_record()
        cls.data=(torch.tensor([[1,2,3,4,5],[6,7,8,0,0]]),torch.tensor([5,3]),torch.tensor([[1.,1.,3.,8.,20.],[1.,1.,9.,0.,0.]],dtype=torch.float64)*838393)

    def test_counts(self):
        self.assertEqual([sum(p.numel() for p in x.parameters()) for x in (self.shared,self.specific)],[715020,715152])
        self.assertEqual([sum(p.numel() for p in cal.parameters()) for times in self.specific.temporal_sets() for cal in times.calibrators.values()],[66]*4)

    def test_two_independent_modules(self):
        self.assertEqual(sum(isinstance(x,ThreeTimes) for x in self.specific.modules()),2)
        self.assertIs(self.shared.temporal_sets()[0],self.shared.temporal_sets()[1])
        self.assertIsNot(*self.specific.temporal_sets())

    def test_equal_values_no_alias(self):self.assertTrue(temporal_aliases(self.specific))

    def test_constructor_rng(self):
        self.assertEqual(self.shared_rng,self.specific_rng);self.assertEqual(self.shared_rng,self.historical_rng)
        before=rng_record();copy.deepcopy(self.specific.times);self.assertEqual(before,rng_record())
        self.assertFalse(torch.cuda.is_initialized())

    def test_initial_scales(self):
        gaps,active=history_gaps(self.data[2],self.data[0]!=0)
        a,b=(times(gaps,active) for times in self.specific.temporal_sets())
        self.assertTrue(all(torch.equal(x,y) and torch.equal(x,torch.ones_like(x)) for x,y in zip(a,b)))

    def test_control_serialization_exact(self):
        def dump(net):
            stream=io.BytesIO();torch.save(net.state_dict(),stream);return stream.getvalue()
        self.assertEqual(list(self.shared.state_dict()),list(self.historical.state_dict()))
        self.assertEqual(dump(self.shared),dump(self.historical))

    def test_common_initialization(self):paired(initial(self.shared),initial(self.specific))

    def test_cpu_forward_parity_and_dropout_rng(self):
        a,b=copy.deepcopy(self.shared).train(),copy.deepcopy(self.specific).train()
        state=capture_rng()
        with patch('experiments.mamba3_three_time.model.three_time_forward',cpu_mixer),patch.object(implementation,'three_time_forward',cpu_mixer):
            x=a.encode_sequence(*self.data);ra=rng_record();restore_rng(state);y=b.encode_sequence(*self.data);rb=rng_record()
        self.assertTrue(torch.equal(x,y));self.assertEqual(ra,rb)

    def test_chain_rule_initial(self):
        for k,v in temporal_chain_cpu(False).items():self.assertTrue(v['passed'],(k,v))

    def test_chain_rule_nonzero(self):
        for k,v in temporal_chain_cpu(True).items():self.assertTrue(v['passed'],(k,v))

    def test_gradcheck_temporal(self):
        cal=nonzero(ThreeTimes('dual').double().calibrators['decay'])
        gaps=torch.tensor([[2e4,9e5]],dtype=torch.float64,requires_grad=True);active=torch.ones_like(gaps,dtype=torch.bool)
        self.assertTrue(torch.autograd.gradcheck(lambda g:cal(g,active),(gaps,),eps=.01,atol=1e-6,rtol=1e-5))

    def test_routing_masks_interventions_roundtrip(self):
        checks={};routing(copy.deepcopy(self.specific),self.data,lambda k,v:checks.update({k:v}),cpu=True)
        self.assertEqual(sorted(checks),c.plan()['required_cases'][3]['required_keys'])
        for k,v in checks.items():self.assertTrue(v['passed'],(k,v))

    def test_dual_same_tensor_and_split(self):
        gap,active=history_gaps(self.data[2],self.data[0]!=0)
        for times in self.specific.temporal_sets():
            d,w,p=times(gap,active);self.assertIs(w,p)
            _,dw,dp=split_dt(torch.ones_like(d),torch.ones_like(d),d,w,p);self.assertIs(dw,dp)

    def test_gap_semantics(self):
        gaps,active=history_gaps(self.data[2],self.data[0]!=0)
        self.assertFalse(bool(active[:,0].any()));self.assertFalse(bool(active[1,3:].any()))
        self.assertEqual(gaps[0,1].item(),0.);self.assertTrue(active[0,1])
        modified=self.data[2].clone();modified[1,3:]=float('nan')
        self.assertTrue(torch.equal(gaps,history_gaps(modified,self.data[0]!=0)[0]))

    def test_fixed_reference(self):
        for times in self.specific.temporal_sets():
            for cal in times.calibrators.values():self.assertEqual(float(cal.reference),838393);self.assertFalse(cal.reference.requires_grad)

    def test_config_parity_cpu(self):
        from recbole.config import Config
        from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
        for v in c.MODES:
            cfg=Config(model=ThreeTimeMamba3Rec,config_dict=c.settings(v,'cpu'));effective_check(cfg,v)
        cfg['learning_rate']=.002
        with self.assertRaises(ValueError):effective_check(cfg,v)

    def test_diagnostics_identical_initial_and_divergence(self):
        net=copy.deepcopy(self.specific);before=rng_record();d=diagnostics(net);self.assertEqual(before,rng_record())
        for m in ('decay','scan'):self.assertEqual(d['divergence'][m]['mean_abs_log_ratio_per_head'],[0.,0.])
        with torch.no_grad():net.layer1_times.calibrators['scan'].last.bias.add_(.1)
        d=diagnostics(net);self.assertGreater(d['divergence']['scan']['all_parameters_l2'],0.)
        self.assertTrue(all(x>0 for x in d['divergence']['scan']['mean_abs_log_ratio_per_head']))

    def test_wrong_variant_rejected(self):
        with self.assertRaises(ValueError):c.settings('triple')
        with self.assertRaises(ValueError):c.settings('layer_specific','bad-device')


if __name__=='__main__':unittest.main()
