"""Reusable synthetic fixtures; no recommendation data loaders."""
import copy
import io
from unittest.mock import patch
import torch
from . import config as c
from .model import LayerTemporalMamba3Rec
from .state import transfer_common
from experiments.mamba3_head_timescales.checks import nonzero
from experiments.mamba3_three_time.confirmation.state import difference,rng_record
from experiments.mamba3_three_time.initialization import seed_all
from experiments.mamba3_timeaware.time_inputs import history_gaps


def fresh(variant,device='cpu',historical=False):
    from recbole.config import Config
    from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
    from experiments.mamba3_three_time.config import SyntheticCatalog
    cfg=Config(model=ThreeTimeMamba3Rec,config_dict=c.settings(variant,device))
    seed_all(314159)
    return (ThreeTimeMamba3Rec if historical else LayerTemporalMamba3Rec)(cfg,SyntheticCatalog()).to(device).eval()


def temporal_aliases(net):
    a,b=net.temporal_sets()
    return all(torch.equal(x,b.state_dict()[k]) and x.data_ptr()!=b.state_dict()[k].data_ptr() for k,x in a.state_dict().items())


def cpu_mixer(mixer,u,decay,write,phase,**kwargs):
    # Differentiable CPU fixture for routing only; actual MIMO kernel is gated on GPU.
    scale=(.2*decay+.3*write+.5*phase).repeat_interleave(u.shape[-1]//2,dim=-1)
    return (u.float()*scale).to(u.dtype)


def routing(net,data,emit,cpu=False):
    import experiments.mamba3_layer_temporal.model as implementation
    sets=net.temporal_sets()
    for times in sets:
        for cal in times.calibrators.values():nonzero(cal)
    device=data[0].device
    before=rng_record();clone=copy.deepcopy(sets[0]);after=rng_record()
    emit('deepcopy_rng',dict(passed=before==after));emit('independent_storage',dict(passed=temporal_aliases(net)))
    del clone
    gaps,active=history_gaps(data[2],data[0]!=0)
    def observed():
        captures=[];outputs={};handles=[]
        for i,times in enumerate(sets):
            def hook(_m,_a,out,index=i):outputs[index]=out
            handles.append(times.register_forward_hook(hook))
        original=cpu_mixer if cpu else implementation.three_time_forward
        def spy(mixer,u,decay,write,phase,**kwargs):
            i=next(j for j,layer in enumerate(net.layers) if layer.mixer is mixer)
            captures.append((i,decay,write,phase))
            return original(mixer,u,decay,write,phase,**kwargs)
        try:
            with patch.object(implementation,'three_time_forward',spy),torch.no_grad():output=net.encode_sequence(*data)
        finally:
            for handle in handles:handle.remove()
        exact=len(captures)==2 and [row[0] for row in captures]==[0,1] and all(all(x is y for x,y in zip(row[1:],outputs[row[0]])) for row in captures)
        return [[v.detach().clone() for v in outputs[i]] for i in (0,1)],output,exact,all(row[2] is row[3] for row in captures)
    a,out,exact,dual=observed()
    emit('routing',dict(passed=exact));emit('dual_same_tensor',dict(passed=dual))
    emit('finite_output',dict(passed=bool(torch.isfinite(out).all())))
    emit('first_padding_neutral',dict(passed=all(torch.equal(v[~active],torch.ones_like(v[~active])) for layer in a for v in layer)))
    with torch.no_grad():sets[0].calibrators['decay'].last.bias.add_(.2)
    b,_,exact,_=observed()
    emit('layer0_intervention',dict(passed=exact and not torch.equal(a[0][0],b[0][0]) and all(torch.equal(x,y) for x,y in zip(a[1],b[1])) and torch.equal(a[0][1],b[0][1])))
    with torch.no_grad():sets[1].calibrators['scan'].last.bias.sub_(.15)
    d,_,exact,_=observed()
    emit('layer1_intervention',dict(passed=exact and not torch.equal(b[1][1],d[1][1]) and all(torch.equal(x,y) for x,y in zip(b[0],d[0])) and torch.equal(b[1][0],d[1][0])))
    zero=torch.zeros_like(gaps);zero_outputs=[times(zero,active) for times in sets]
    emit('active_zero',dict(passed=bool(active[:,1:].any()) and all(bool((v[active]!=1).any()) for layer in zero_outputs for v in layer)))
    emit('fixed_reference',dict(passed=all(float(cal.reference)==838393 and not cal.reference.requires_grad for times in sets for cal in times.calibrators.values())))
    emit('scale_bounds',dict(passed=all(bool(((v>=.5)&(v<=2)).all()) for layer in d for v in layer)))
    from recbole.data.interaction import Interaction
    interaction=Interaction({net.ITEM_SEQ:data[0],net.ITEM_SEQ_LEN:data[1],net.time_sequence_field:data[2],c.settings('layer_specific')['TIME_FIELD']:torch.ones(len(data[0]),device=device)})
    original=cpu_mixer if cpu else implementation.three_time_forward
    with patch.object(implementation,'three_time_forward',original),torch.no_grad():
        first=net._encode(interaction);interaction[c.settings('layer_specific')['TIME_FIELD']].fill_(9e14);second=net._encode(interaction)
    emit('target_timestamp_ignored',difference(first,second))
    stream=io.BytesIO();torch.save(net.state_dict(),stream);stream.seek(0)
    state=torch.load(stream,map_location=device,weights_only=True);net.load_state_dict(state,strict=True)
    emit('roundtrip',dict(passed=all(torch.equal(v,state[k]) for k,v in net.state_dict().items())))
    emit('hook_cleanup',dict(passed=all(not times._forward_hooks for times in sets)))
    emit('mimo_rank_chunk',dict(passed=all(layer.mixer.is_mimo and layer.mixer.mimo_rank==4 and layer.mixer.chunk_size==8 for layer in net.layers)))
    emit('parameter_count',dict(passed=sum(p.numel() for p in net.parameters())==715152))


def temporal_chain_cpu(nontrivial=True):
    from experiments.mamba3_three_time.calibrators import ThreeTimes
    shared=ThreeTimes('dual').double()
    if nontrivial:
        for cal in shared.calibrators.values():nonzero(cal)
    left,right=copy.deepcopy(shared),copy.deepcopy(shared)
    gaps=torch.tensor([[0.,8383.93,838393.,83839300.]],dtype=torch.float64);active=torch.tensor([[False,True,True,True]])
    def objective(first,second):
        a,b=first(gaps,active),second(gaps,active)
        return sum((x*y*torch.tensor([.7,1.3],dtype=torch.float64)).sum() for x,y in zip(a,b))
    x,y=objective(shared,shared),objective(left,right);x.backward();y.backward()
    checks={'output':difference(x,y)}
    for k,v in shared.named_parameters():
        a,b=dict(left.named_parameters())[k],dict(right.named_parameters())[k]
        checks[k]=difference(v.grad,a.grad+b.grad)
    return checks
