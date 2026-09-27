import math
import pytest
import torch
from experiments.mamba3_three_time.fixtures import kernel_inputs
from experiments.mamba3_three_time.reference import recurrence, intermediates, rotary


@pytest.mark.parametrize("architecture",["SISO","MIMO"])
def test_independent_path_gradcheck(architecture):
    x = kernel_inputs(architecture,length=3,device="cpu",dtype=torch.float64,tiny=True)
    keys = ("adt","dw","dp","trap","angles")
    def calculate(*values):
        return recurrence(**{**x,**dict(zip(keys,values))})
    assert torch.autograd.gradcheck(calculate,tuple(x[k] for k in keys),eps=1e-6,atol=1e-5,rtol=1e-3)


def test_interventions():
    x = kernel_inputs("SISO",length=3,device="cpu",dtype=torch.float64,tiny=True)
    fields = [x[k] for k in ("adt","dw","dp","trap","angles")]
    original = intermediates(*fields)
    phase = intermediates(fields[0],fields[1],fields[2]*1.3,*fields[3:])
    write = intermediates(fields[0],fields[1]*1.3,*fields[2:])
    decay = intermediates(fields[0]*1.3,*fields[1:])
    assert not torch.equal(original["theta"],phase["theta"])
    for key in ("gamma","beta","decay"):
        assert torch.equal(original[key],phase[key])
    assert not torch.equal(original["gamma"],write["gamma"])
    assert torch.equal(original["theta"],write["theta"])
    assert torch.equal(original["decay"],write["decay"])
    assert not torch.equal(original["decay"],decay["decay"])
    assert torch.equal(original["theta"],decay["theta"])
    # ADT also weights beta/previous input in the recurrence, not only old state.
    assert not torch.equal(original["decay"]*original["beta"],decay["decay"]*decay["beta"])


@pytest.mark.parametrize("architecture",["SISO","MIMO"])
def test_recurrence_matches_expanded_causal_sum(architecture):
    x = kernel_inputs(architecture,length=4,device="cpu",dtype=torch.float64,tiny=True)
    actual = recurrence(**x)
    mimo = architecture == "MIMO"
    q,k = x["q"],x["k"]
    if mimo:
        q,k = q.permute(0,1,3,2,4),k.permute(0,1,3,2,4)
        qb,kb = x["qb"],x["kb"]
    else:
        q,k,qb,kb = q.unsqueeze(-2),k.unsqueeze(-2),x["qb"].unsqueeze(1),x["kb"].unsqueeze(1)
    q,k = q.repeat_interleave(2,2)+qb,k.repeat_interleave(2,2)+kb
    info = intermediates(*(x[k] for k in ("adt","dw","dp","trap","angles")))
    qr,kr = rotary(q,info["theta"],mimo),rotary(k,info["theta"],mimo)
    v = x["v"].unsqueeze(-2)
    if mimo:
        v = v*x["mv"]
    expected = []
    for t in range(4):
        y = info["gamma"][:,:,t,None,None]*torch.einsum("bhrn,bhsn,bhsp->bhrp",q[:,t],k[:,t],v[:,t])
        for source in range(t):
            coefficient = (info["gamma"][:,:,source]+info["beta"][:,:,source+1]) * x["adt"][:,:,source+1:t+1].sum(-1).exp()
            y = y+coefficient[...,None,None]*torch.einsum("bhrn,bhsn,bhsp->bhrp",qr[:,t],kr[:,source],v[:,source])
        y = y+x["d"][None,:,None,None]*v[:,t]
        gate = x["z"][:,t].unsqueeze(-2)*(x["mz"] if mimo else 1)
        y = y*torch.nn.functional.silu(gate)
        expected.append((y*x["mo"]).sum(-2) if mimo else y.squeeze(-2))
    torch.testing.assert_close(actual,torch.stack(expected,1),atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize("architecture",["SISO","MIMO"])
def test_mod_boundary_finite_and_causal(architecture):
    x = kernel_inputs(architecture,length=4,batch=2,device="cpu",dtype=torch.float64,tiny=True)
    x["dp"] = torch.full_like(x["dp"],2/math.tanh(.1),requires_grad=True)
    x["angles"] = torch.full_like(x["angles"],.1,requires_grad=True)
    out = recurrence(**x)
    out[0,:2].sum().backward()
    assert torch.isfinite(out).all()
    for value in x.values():
        assert value.grad is not None and torch.isfinite(value.grad).all()
    assert torch.count_nonzero(x["v"].grad[0,2:]) == 0
    assert torch.count_nonzero(x["v"].grad[1]) == 0
