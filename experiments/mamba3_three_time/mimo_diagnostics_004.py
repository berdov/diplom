"""Fixed MIMO L15/L50 and D-only diagnostics, original kernels/adapter."""

import torch
from .fixtures import kernel_inputs, clone_inputs, kernel_output, scalar_loss
from .suites import comparisons
from .evidence import case, compare, tensor_records
from .numerics_004 import packed, cotangent, loss_report, d_oracles


def fixture(length,tied,d_only=False):
    x=kernel_inputs("MIMO",length=length,tied=tied)
    with torch.no_grad():
        if not tied:
            variation=torch.linspace(.7,1.3,x['dw'].numel(),device=x['dw'].device).reshape_as(x['dw'])
            x['adt'].mul_(variation)
            x['dw'].mul_(variation.flip(-1))
            x['dp'].mul_(1.8-variation)
        if d_only:
            for key in ('q','k','qb','kb'):x[key].zero_()
    return x


def measure(source,backend,go=None):
    x=clone_inputs(source,float_reference=backend=='reference')
    y=kernel_output(x,'MIMO',official=backend=='official',reference=backend=='reference')
    loss=scalar_loss(y)
    incoming=torch.autograd.grad(loss,y,retain_graph=True)[0] if go is None else go.to(y.dtype)
    unique={}
    seen=set()
    for k,v in x.items():
        if id(v) not in seen:unique[k]=v;seen.add(id(v))
    gradients=torch.autograd.grad(y,tuple(unique.values()),grad_outputs=incoming,allow_unused=True)
    values={'output':y.detach(),'loss':loss.detach()}
    values.update({'gradient:'+k:g.detach() if g is not None else None for k,g in zip(unique,gradients)})
    return values,incoming.detach()


def report(length,tied,profile,d_only=False):
    source=fixture(length,tied,d_only)
    backends=['local','reference']+(['official'] if tied else [])
    legacy={b:measure(source,b) for b in backends}
    reference,ref_go=legacy['reference']
    records={}
    checks={}
    for backend in backends:
        values,incoming=legacy[backend]
        checks[backend+':legacy_measured']=dict(passed=all(v is not None and bool(torch.isfinite(v).all()) for v in values.values()))
        records[backend]=dict(required=False,backend='upstream '+backend,
            legacy_comparisons=comparisons(values,reference,profile),
            legacy_D_gradient=packed(values['gradient:d']),legacy_loss=packed(values['loss']),
            legacy_grad_output=packed(incoming),outputs=packed(values['output']),
            loss_diagnostics=loss_report(values['output'],reference['output'],profile),
            legacy_incoming_gradient_vs_reference=compare(incoming,ref_go))
    shared=[]
    for kind in ('signed','nonnegative'):
        go,meta=cotangent(reference['output'].shape,kind,reference['output'].device)
        values={b:measure(source,b,go)[0] for b in backends}
        oracles=d_oracles(source,go)
        row=dict(kind=kind,cotangent=meta,required=False,backend='upstream',
            D_oracles={k:packed(v) for k,v in oracles.items()},backends={})
        for backend,v in values.items():
            checks[backend+':'+kind+':all_gradients']=dict(passed=all(w is not None and bool(torch.isfinite(w).all()) for w in v.values()))
            all_checks=comparisons(v,values['reference'],profile)
            row['backends'][backend]=dict(comparisons={k:w for k,w in all_checks.items() if k.startswith('gradient:')},
                output_discrepancy=all_checks['output'],loss_not_a_vjp_check=all_checks['loss'],
                D_actual=packed(v['gradient:d']),D_oracle_comparisons={k:compare(v['gradient:d'],w,
                    atol=profile['gradient_atol'],rtol=profile['gradient_rtol'],
                    relative_norm_limit=profile['gradient_relative_norm_limit']) for k,w in oracles.items()})
        if tied:row['split_vs_official']=comparisons(values['local'],values['official'])
        shared.append(row)
    incoming_effect={}
    for backend in ('local','official') if tied else ('local',):
        actual,go=legacy[backend]
        reference_common=measure(source,'reference',go)[0]
        actual_ref_go=measure(source,backend,ref_go)[0]
        incoming_effect[backend]=dict(required=False,
            reference_with_kernel_incoming_vs_reference_own=comparisons(reference_common,reference,profile),
            kernel_own_vs_kernel_reference_incoming=comparisons(actual,actual_ref_go,profile),
            same_kernel_incoming_reference_vs_kernel=comparisons(actual,reference_common,profile),
            note='Includes scalar-loss output dependence AND incoming bf16 rounding; not only output dependence.',
            analytic_unrounded_gradient_delta=packed(.06*(actual['output'].double()-reference['output'].double())/reference['output'].numel()))
    row=case('mimo_forensics','fixed fixtures; shared VJP and independent D coefficient',checks)
    row.update(backend='upstream',length=length,tied=tied,d_only=d_only,
        input_tensors=tensor_records(source),legacy=records,common_vjp=shared,
        incoming_gradient_effect=incoming_effect,D_meaning='skip connection weight, not dw/dp',
        oracle_limitations='Rounded coefficient uses continuous/STE convention, not derivative of discrete casts. '
            'Pinned tanh/reduction may differ from PyTorch; unavailable equivalence remains unproven.',
        training_authorized=False)
    return row
