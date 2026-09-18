import ast
import copy
from pathlib import Path

import pytest
import torch

from experiments.mamba3_input_time.adapter import InputAdapter, ADAPTER_COUNTS, MODES, log_interval
from experiments.mamba3_input_time.checks import all_pass, fixture, nonzero, reduction, reference_check, semantics
from experiments.mamba3_input_time.diagnostics import InputDiagnostics
from experiments.mamba3_input_time.provenance import rng_hash
from experiments.mamba3_timeaware.time_inputs import history_gaps


@pytest.mark.parametrize('mode',MODES)
def test_counts_identity_contract(mode):
    u,times,valid = fixture()
    snapshots = [x.clone() for x in (u,times,valid)]
    model = InputAdapter(mode)
    assert sum(p.numel() for p in model.parameters()) == ADAPTER_COUNTS[mode]
    result = model(u,times,valid)
    assert result.shape == u.shape and result.dtype == torch.float32
    torch.testing.assert_close(result,u,atol=0,rtol=0)
    for a,b in zip((u,times,valid),snapshots):
        torch.testing.assert_close(a,b,equal_nan=True,atol=0,rtol=0)
    with pytest.raises(ValueError):
        model(u,times.float(),valid)
    broken = valid.clone()
    broken[0,1] = False
    with pytest.raises(ValueError):
        model(u,times,broken)
    with pytest.raises(ValueError):
        model(u,times,torch.zeros_like(valid))


def test_trained_like_causality_and_masking():
    checks = semantics()
    assert all_pass(checks), {k:v for k,v in checks.items() if not all_pass(v)}


def test_content_time_reduction_gradients():
    assert all_pass(reduction())


def test_independent_loop_reference():
    checks = reference_check()
    assert all_pass(checks), {k:v for k,v in checks.items() if not all_pass(v)}


def test_masked_nonfinite_values_are_not_keys():
    u,times,valid = fixture()
    a = InputAdapter('attention_time')
    nonzero(a)
    expected = a(u,times,valid)
    u[~valid] = float('nan')
    actual = a(u,times,valid)
    torch.testing.assert_close(actual[valid],expected[valid],atol=0,rtol=0)


def test_float64_attention_gradcheck():
    model = InputAdapter('attention_time').double()
    nonzero(model)
    gen = torch.Generator().manual_seed(12)
    u = torch.randn(1,3,64,generator=gen,dtype=torch.float64,requires_grad=True)
    times = torch.tensor([[1.6e12,1.6e12+5,1.6e12+1000000]],dtype=torch.float64)
    valid = torch.ones(1,3,dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda x:model(x,times,valid),(u,),fast_mode=True)


def test_float64_precision_and_overflow():
    times = torch.tensor([[1.6e12,1.6e12+5,1.6e12+10]],dtype=torch.float64)
    valid = torch.ones_like(times,dtype=torch.bool)
    gaps,active = history_gaps(times,valid)
    assert gaps.tolist()==[[0,5,5]] and active.tolist()==[[False,True,True]]
    tau = log_interval(torch.tensor([0,5,1e308],dtype=torch.float64))
    assert torch.isfinite(tau).all() and tau[0]==0 and tau[1]>0


@pytest.mark.parametrize('mode',MODES[1:])
def test_output_then_inner_gradients_and_updates(mode):
    torch.manual_seed(71)
    model = InputAdapter(mode)
    u,times,valid = fixture()
    optimizer = torch.optim.Adam(model.parameters(),lr=.001)
    before = copy.deepcopy(model.state_dict())
    for step in range(4):
        optimizer.zero_grad()
        loss = model(u,times,valid)[valid].square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if step==0 and mode.startswith('attention_'):
            assert model.q.weight.grad.abs().sum()==0
            assert model.out.weight.grad.abs().sum()>0
        if step==3:
            assert all(p.grad.abs().sum()>0 for p in model.parameters())
        optimizer.step()
    assert all(not torch.equal(before[k],p) for k,p in model.state_dict().items())


@pytest.mark.parametrize('mode',MODES)
def test_diagnostics_output_and_rng_parity(mode):
    u,times,valid = fixture()
    model = InputAdapter(mode)
    nonzero(model)
    baseline = model(u,times,valid)
    before = rng_hash()
    output,details = model(u,times,valid,details=True)
    collector = InputDiagnostics(mode)
    _,active = history_gaps(times,valid)
    scales = torch.ones(*valid.shape,2)
    collector.update(u,valid,details,scales,scales,active)
    result = collector.result()
    assert rng_hash()==before
    torch.testing.assert_close(output,baseline,atol=0,rtol=0)
    assert result['attention_heads']==4 and result['mamba_temporal_heads']==2
    assert result['statistics']['delta_norm_ratio']['count']==int(valid.sum())


def test_encoder_reads_only_history_not_targets():
    path = Path(__file__).resolve().parents[2]/'mamba3_timeaware/model.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_encode')
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
    class Encoder:
        ITEM_SEQ,ITEM_SEQ_LEN,time_sequence_field = 'ids','lengths','times'
        def forward(self,*args):
            return args
    class HistoryOnly(dict):
        def __getitem__(self,key):
            assert key in ('ids','lengths','times')
            return super().__getitem__(key)
    result = namespace['_encode'](Encoder(),HistoryOnly(ids=1,lengths=2,times=3,target_item=99,target_timestamp=1e30))
    assert result==(1,2,3)


def test_quite_pinned_broadcast_reproduction():
    # Reproduce the inspected broadcasting expression, not third-party classes/training code.
    scores = torch.zeros(1,1,3,3)
    mask = torch.tensor([[[1],[1],[0]]])
    weights = scores.masked_fill(mask.unsqueeze(1)==0,-10000).softmax(-1)
    values = torch.tensor([[[[0.,0.],[0.,0.],[1.,2.]]]])
    changed = values.clone()
    changed[:,:,2] *= 10
    assert mask.unsqueeze(1).shape==(1,1,3,1)
    delta = (weights@changed-weights@values)[0,0,0]
    torch.testing.assert_close(delta,torch.tensor([3.,6.]))
    correct = scores.masked_fill(~torch.tensor([True,True,False])[None,None,None,:],-torch.inf).softmax(-1)
    torch.testing.assert_close((correct@changed)[0,0,0],(correct@values)[0,0,0],atol=0,rtol=0)
    # The effect also survives the inspected residual + LayerNorm path for noncollinear values.
    tokens = torch.tensor([[[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.]]]])
    alternative = tokens.clone()
    alternative[:,:,2] = torch.tensor([0.,0.,0.,10.])
    def query_embedding(w,x):
        return torch.nn.functional.layer_norm(x+w@x,(4,))[0,0,0]
    assert (query_embedding(weights,tokens)-query_embedding(weights,alternative)).abs().max()>1
    torch.testing.assert_close(query_embedding(correct,tokens),query_embedding(correct,alternative),atol=0,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Real pinned Mamba3/A100 identity is a mandatory allocation gate, not CPU evidence')
def test_full_model_gpu_is_not_run_by_cpu_suite():
    pytest.skip('GPU gates execute only inside the single guarded Slurm pipeline')
