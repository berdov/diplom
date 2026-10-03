"""Small analytic gap grid, no dataset pass or checkpoint loading."""
import torch
from . import config as c


@torch.no_grad()
def diagnostics(model):
    sets=model.temporal_sets();device=next(model.parameters()).device
    grid=torch.tensor(c.plan()['diagnostic_grid'],device=device,dtype=torch.float64).reshape(1,-1)
    active=torch.ones_like(grid,dtype=torch.bool);layers=[];curves=[]
    for times in sets:
        layer={};values={}
        for name,cal in times.calibrators.items():
            curve=cal(grid*838393,active)[0].double();values[name]=curve
            layer[name]=dict(gaps_over_R0=c.plan()['diagnostic_grid'],scale_by_gap_and_head=curve.cpu().tolist(),
                at_reference=curve[5].cpu().tolist(),near_lower=(curve<.51).double().mean(0).cpu().tolist(),
                near_upper=(curve>1.99).double().mean(0).cpu().tolist(),reference_ms=float(cal.reference))
        layers.append(layer);curves.append(values)
    divergence={}
    for name in ('decay','scan'):
        delta=(curves[0][name]/curves[1][name]).log().abs()
        params={};all_a=[];all_b=[]
        for key,a in sets[0].calibrators[name].named_parameters():
            b=dict(sets[1].calibrators[name].named_parameters())[key];a=a.double();b=b.double()
            l2=(a-b).norm().item();ref=a.norm().item()
            params[key]=dict(l2=l2,reference_l2=ref,relative_l2=l2/ref if ref else None)
            all_a.append(a.reshape(-1));all_b.append(b.reshape(-1))
        a,b=torch.cat(all_a),torch.cat(all_b);l2=(a-b).norm().item();ref=a.norm().item()
        divergence[name]=dict(mean_abs_log_ratio_per_head=delta.mean(0).cpu().tolist(),max_abs_log_ratio_per_head=delta.max(0).values.cpu().tolist(),
            parameter_distances=params,all_parameters_l2=l2,reference_parameters_l2=ref,relative_parameters_l2=l2/ref if ref else None)
    return dict(temporal_sharing=model.temporal_sharing,layers=layers,divergence=divergence,
        relative_l2_denominator='layer0 norm; null if zero',near_bounds='scale < 0.51 / scale > 1.99',
        scope='Analytic active-gap grid, not dataset fractions; no layer semantics inferred')
