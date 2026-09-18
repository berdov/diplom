"""Mandatory synthetic A100 gates A-F, before any recommendation fit."""
import gc
import json
import os
import time
import traceback

from .adapter import COUNTS, MODES
from .checks import all_pass, compare, nonzero, reduction, reference_check, semantics
from .config import GPU_EVIDENCE, RUNTIME, plan, settings
from .provenance import atomic_json, exclusive, now, require_submission, save_state_dict, PIN, backbone_hash, rng_hash, tensor_hash


class Catalog:
    def num(self, field):
        return 7112 if field == 'item_id' else 23952


def make_model(mode, training=True):
    from recbole.config import Config
    from .model import InputMamba3Rec
    task = next(t for t in plan()['tasks'] if t['mode'] == mode)
    return InputMamba3Rec(Config(model=InputMamba3Rec, config_dict=settings(task)), Catalog()).cuda().train(training)


def batch_for(batch_size, length):
    import torch
    ids = torch.randint(1,7112,(batch_size,length),device='cuda')
    lengths = torch.randint(1,length+1,(batch_size,),device='cuda')
    lengths[0] = length
    if batch_size > 1:
        lengths[-1] = 1
    ids[torch.arange(length,device='cuda')[None,:] >= lengths[:,None]] = 0
    times = torch.randint(1,2000000,(batch_size,length),device='cuda').double().cumsum(1)+1.6e12
    times[:,1] = times[:,0]
    times[ids==0] = float('nan')
    return ids,lengths,times,torch.randint(1,7112,(batch_size,),device='cuda')


def side(model, batch):
    import torch
    observed = []
    # Observation only in synthetic evidence, never a hook to pass/replace model inputs.
    def capture(module, args, output):
        output.retain_grad()
        observed.append(output)
    handle = model.item_embedding.register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        torch.manual_seed(2026)
        ids,lengths,times,targets = batch
        encoded = model(ids,lengths,times)
        scores = encoded @ model.item_embedding.weight.T
        loss = model.loss_fct(scores,targets)
        loss.backward()
        grads = {k:p.grad.detach().clone() if p.grad is not None else None for k,p in model.named_parameters()}
        return dict(encoded=encoded.detach(), candidate_scores=scores[:,[1,3,19,7111]].detach(),
                    loss=loss.detach(), input_gradient=observed[0].grad.detach(), gradients=grads)
    finally:
        handle.remove()


def compare_sides(a,b):
    checks = {key:compare(a[key],b[key]) for key in ('encoded','candidate_scores','loss','input_gradient')}
    checks.update({'gradient:'+key:compare(value,b['gradients'].get(key))
                   for key,value in a['gradients'].items() if not key.startswith('input_adapter.')})
    return checks


def identity_case(training,length):
    import torch
    from recbole.config import Config
    from experiments.mamba3_time_mechanisms.model import MechanismMamba3Rec
    torch.manual_seed(811)
    cfg = Config(model=MechanismMamba3Rec, config_dict=settings(plan()['tasks'][0]))
    reference = MechanismMamba3Rec(cfg,Catalog()).cuda().train(training)
    nonzero(reference.mechanisms)
    batch = batch_for(3,length)
    expected = side(reference,batch)
    checks = {}
    for mode in MODES:
        model = make_model(mode,training)
        model.load_state_dict({**model.state_dict(),**reference.state_dict()},strict=True)
        checks[mode] = compare_sides(expected,side(model,batch))
        del model
    return dict(training=training,length=length,checks=checks,passed=all_pass(checks))


def initialization_gate():
    from recbole.utils import init_seed
    rows = []
    for mode in MODES:
        init_seed(2026,True)
        model = make_model(mode)
        rows.append(dict(mode=mode,total=sum(p.numel() for p in model.parameters()),
                         backbone=backbone_hash(model),rng=rng_hash(),
                         adapter=tensor_hash(model.input_adapter.state_dict()),
                         attention_common=tensor_hash(model.input_adapter.common_state()) if mode.startswith('attention_') else None))
        del model
    if (len({r['backbone'] for r in rows})!=1 or len({r['rng'] for r in rows})!=1
            or rows[2]['attention_common']!=rows[3]['attention_common']
            or {r['mode']:r['total'] for r in rows}!=COUNTS):
        raise ValueError('GPU full-model initialization parity failed')
    return rows


def update_roundtrip():
    import torch
    checks = {}
    batch = batch_for(3,50)
    for mode in MODES:
        model = make_model(mode)
        optimizer = torch.optim.Adam(model.parameters(),lr=.001)
        before = {k:p.detach().clone() for k,p in model.input_adapter.named_parameters()}
        for step in range(3):
            side(model,batch)
            checks[f'{mode}:finite_step{step}'] = dict(passed=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
            if step == 2:
                checks[f'{mode}:inner_gradient'] = dict(passed=all(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.input_adapter.parameters()))
            optimizer.step()
        checks[mode+':updated'] = dict(passed=all(not torch.equal(before[k],p) for k,p in model.input_adapter.named_parameters()))
        path = RUNTIME/'gpu_gate'/f'{mode}_roundtrip.pth'
        if path.exists():
            raise FileExistsError(path)
        digest = save_state_dict(model,path)
        clone = make_model(mode)
        clone.load_state_dict(torch.load(path,weights_only=True,map_location='cuda'),strict=True)
        checks[mode+':roundtrip'] = dict(state=compare_sides(side(model,batch),side(clone,batch)),
                                       checksum=dict(passed=tensor_hash(model.state_dict())==tensor_hash(clone.state_dict()),sha256=digest))
        del model,clone,optimizer
    return checks


def memory_gate():
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = make_model('attention_time')
    batch = batch_for(2048,50)
    torch.cuda.synchronize()
    start = time.perf_counter()
    ids,lengths,times,targets = batch
    encoded = model(ids,lengths,times)
    loss = model.loss_fct(encoded@model.item_embedding.weight.T,targets)
    loss.backward()
    torch.cuda.synchronize()
    result = dict(passed=bool(torch.isfinite(loss)) and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                  batch_size=2048,length=50,mode='attention_time',loss=float(loss.detach()),
                  seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),automatic_batch_reduction=False)
    return result


def require_gpu_evidence(evidence,manifest):
    if (evidence.get('status')!='PASS' or evidence.get('source_hash')!=manifest['source_hash']
            or evidence.get('execution_commit')!=os.environ['RUN_COMMIT']
            or evidence.get('job_id')!=os.environ['SLURM_JOB_ID'] or evidence.get('pinned_mamba')!=PIN
            or (evidence.get('atol'),evidence.get('rtol'))!=(1e-6,1e-5)
            or evidence.get('test_evaluation_count')!=0):
        raise ValueError('Current-execution GPU PASS required')
    cases = evidence.get('identity_cases',[])
    if len(cases)!=4 or {(r['training'],r['length']) for r in cases}!={(t,l) for t in (False,True) for l in (50,64)}:
        raise ValueError('Incomplete identity GPU evidence')
    if any(set(r['checks'])!=set(MODES) or not r['passed'] or not all_pass(r['checks']) for r in cases):
        raise ValueError('Identity GPU gate failed')
    if set(evidence.get('checks',{}))!={'B_semantics','C_reduction','D_loop_reference','E_updates_roundtrip','F_memory'} or not all_pass(evidence['checks']):
        raise ValueError('Incomplete/failed GPU checks B-F')
    if (evidence['checks']['F_memory']['batch_size'],evidence['checks']['F_memory']['length'])!=(2048,50):
        raise ValueError('Memory gate size differs')


def main():
    manifest = require_submission()
    if GPU_EVIDENCE.exists():
        raise FileExistsError('GPU evidence exists')
    exclusive(RUNTIME/'gpu_gate/gate.lock',dict(started_at=now(),stage='GPU_GATE'))
    result = dict(status='RUNNING',stage='GPU_GATE',source_hash=manifest['source_hash'],pinned_mamba=PIN,
                  atol=1e-6,rtol=1e-5,synthetic_only=True,identity_cases=[],checks={},test_evaluation_count=0,
                  execution_commit=os.environ['RUN_COMMIT'],job_id=os.environ['SLURM_JOB_ID'],
                  node=os.environ.get('SLURMD_NODENAME'),started_at=now())
    atomic_json(GPU_EVIDENCE,result)
    try:
        import torch
        from .preflight import runtime_check
        result['runtime'] = runtime_check(require_cuda=True)
        result['initialization'] = initialization_gate()
        for training in (False,True):
            for length in (50,64):
                row = identity_case(training,length)
                result['identity_cases'].append(row)
                atomic_json(GPU_EVIDENCE,result)
                if not row['passed']:
                    raise ValueError(f'Identity GPU gate failed: train={training}, L={length}')
        stages = [('B_semantics',lambda:semantics('cuda')),('C_reduction',lambda:reduction('cuda')),
                  ('D_loop_reference',lambda:reference_check('cuda')),('E_updates_roundtrip',update_roundtrip),('F_memory',memory_gate)]
        for name,call in stages:
            require_submission()
            result['stage'] = name
            atomic_json(GPU_EVIDENCE,result)
            result['checks'][name] = call()
            atomic_json(GPU_EVIDENCE,result)
            if not all_pass(result['checks'][name]):
                raise ValueError('GPU gate failed: '+name)
        result.update(status='PASS',stage='COMPLETED')
        require_gpu_evidence(result,manifest)
    except BaseException as exc:
        result.update(status='GPU_GATE_FAIL',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        result['finished_at'] = now()
        atomic_json(GPU_EVIDENCE,result)


if __name__ == '__main__':
    main()
