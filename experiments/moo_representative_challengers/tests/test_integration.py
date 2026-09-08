"""Shared five-task training loop, HVI graph replay, and selection regressions."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.moo_representative_challengers import training
from experiments.moo_representative_challengers.common import load_config, select_operating_point, verify_historical_inputs
from experiments.moo_representative_challengers.methods import most, phn_hvi
from experiments.moo_8families.pareto_models.phn import PHNAdapterTiM4Rec


class ToyModel(nn.Module):
    def __init__(self, adapter=False):
        super().__init__()
        self.backbone=nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Dropout(.1))
        self.rank=nn.Linear(8,1)
        self.click_head=nn.Linear(8,1); self.long_view_head=nn.Linear(8,1)
        self.like_head=nn.Linear(8,1); self.profile_enter_head=nn.Linear(8,1)
        self.adapter=adapter; self.hidden_size=8; self.adapter_scale=.1
        if adapter:
            self.preference_hypernetwork=nn.Sequential(nn.Linear(5,64),nn.ReLU(),nn.Linear(64,16))
            nn.init.zeros_(self.preference_hypernetwork[-1].weight); nn.init.zeros_(self.preference_hypernetwork[-1].bias)
            self.register_buffer('current_preference',torch.ones(5)/5)
    set_preference=PHNAdapterTiM4Rec.set_preference
    conditioned_representation=PHNAdapterTiM4Rec.conditioned_representation
    def preference(self,reference=None):
        return PHNAdapterTiM4Rec.preference(self,reference).clone()
    def losses(self,x):
        h=self.backbone(x)
        if self.adapter: h=self.conditioned_representation(h)
        return torch.stack([((head(h).squeeze()-x[:,i%4])**2).mean()+.1 for i,head in enumerate(
            [self.rank,self.click_head,self.long_view_head,self.like_head,self.profile_enter_head])])


@pytest.mark.parametrize('inverse_priority',[False,True])
def test_hvi_rng_replay_matches_joint_adapter_gradients(inverse_priority):
    torch.manual_seed(2026)
    model=ToyModel(True)
    # Reach a nonconstant adapter using an actual differentiable step from identical zero-init.
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    model.set_preference([.6,.1,.1,.1,.1])
    x=torch.randn(9,4)
    model.losses(x).sum().backward(); optimizer.step(); optimizer.zero_grad()
    other=deepcopy(model)
    prefs=phn_hvi.preference_samples(np.random.default_rng(2026))
    state=torch.get_rng_state()
    points=[]
    for r in prefs:
        model.set_preference(r); points.append(model.losses(x))
    losses=torch.stack(points)
    w,_=phn_hvi.hv_weights(losses.detach().numpy(),[1.5]*5)
    loss_rays=phn_hvi.inverse_priority_rays(prefs) if inverse_priority else prefs
    scalar=phn_hvi.surrogate(losses,loss_rays,w); scalar.backward()
    final_rng=torch.get_rng_state()
    torch.set_rng_state(state)
    value,_,_=phn_hvi.replay_backward(other,prefs,lambda:other.losses(x),[1.5]*5,inverse_priority=inverse_priority)
    torch.testing.assert_close(torch.get_rng_state(),final_rng)
    assert abs(value-float(scalar.detach()))<2e-5
    for a,b in zip(model.parameters(),other.parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=1e-6,rtol=1e-5)
    model.eval()
    model.set_preference(prefs[0]); p1=model.preference_hypernetwork(model.preference()).detach()
    model.set_preference(prefs[1]); p2=model.preference_hypernetwork(model.preference()).detach()
    assert (p1-p2).norm()>0
    assert model.preference_hypernetwork[0].weight.grad.norm()>0


@pytest.mark.parametrize('method',['ferero','most','phn_hvi'])
def test_actual_training_loop_five_heads_and_shared_updates(method):
    cfg=load_config(); mc=cfg['methods'][method]
    count=6 if method=='ferero' else 3 if method=='most' else 1
    models=[]
    for i in range(count):
        torch.manual_seed(2026+i); models.append(ToyModel(method=='phn_hvi'))
    snapshots=[training.parameter_snapshot(m) for m in models]
    optimizers=[torch.optim.Adam(m.parameters(),lr=.001) for m in models]
    helpers=SimpleNamespace(
        task_losses=lambda model,interaction,*a,**kw:{'normalized_task_vector':model.losses(interaction)},
        shared_parameter_entries=lambda m,_:[SimpleNamespace(parameter=p) for p in m.backbone.parameters()],
        interaction_from_batch=lambda b:b)
    preferences=[{'id':str(i),'weights':[.6,.1,.1,.1,.1]} for i in range(6)]
    result=training.train_epoch(method,models,optimizers,[torch.randn(9,4) for _ in range(3)],{}, {},[1]*5,mc,
        preferences,most.MosT(**mc['solver']) if method=='most' else None,np.random.default_rng(2026),helpers,max_batches=3)
    assert result['batches']==3 and result['examples']==27
    for m,b in zip(models,snapshots):
        assert training.update_deltas(m,b)['backbone']>0
    if method=='most': assert len(set(training.parameter_hash(m) for m in models))==3
    for head in ['click_head','long_view_head','like_head','profile_enter_head']:
        assert any(g.get(head,0)>0 for g in result['gradient_norm_max_by_model'])


def test_selection_is_scalarized_not_ndcg_oracle():
    def point(index,ndcg,bce):
        return {'solution_index':index,'metrics':{'NDCG@10':ndcg},'auxiliary_validation':{
            k:{'bce_loss':bce} for k in ['is_click','long_view','is_like','is_profile_enter']}}
    # Highest NDCG has very poor auxiliary BCE and must lose under frozen score.
    points=[point(0,.1,1.8),point(1,.05,.1),point(2,.03,.2)]
    assert select_operating_point(points,'most',load_config())['solution_index']==1
    same=[point(2,.1,.2),point(1,.1,.2)]
    assert select_operating_point(same,'most',load_config())['solution_index']==1


def test_historical_input_regression():
    assert verify_historical_inputs()['files_verified']>20


def test_source_certificate_excludes_isolated_runtime_packages():
    import tempfile
    from pathlib import Path
    from experiments.moo_representative_challengers.common import HERE,source_digest
    runtime=HERE/'runtime'; runtime.mkdir(exist_ok=True)
    before=source_digest()
    with tempfile.TemporaryDirectory(dir=runtime) as directory:
        (Path(directory)/'third_party.py').write_text('# Installed dependency, not experiment source\n')
        assert source_digest()==before
