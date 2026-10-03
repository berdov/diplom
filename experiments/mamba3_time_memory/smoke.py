"""Three synthetic Adam steps at full frozen shapes; no recommendation data."""
import io
import traceback
import torch
from . import config as c
from .checks import fresh
from .memory import select
from .provenance import identity, require_stage, runtime
from experiments.mamba3_mimo_time.records import create, update, now
from experiments.mamba3_mimo_time.provenance import assert_upstream
from experiments.mamba3_mimo_time.admission import histories


def main():
    base = identity()
    record = dict(**base,status='RUNNING',targeted_gate_sha256=require_stage(c.GATE,base),scientific_fits=0,
                  batch=2048,history_length=50,kernel_length=56,steps_per_mode=3,rows=[])
    create(c.SMOKE,record)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        record['runtime'] = runtime(True)
        data = histories(50,batch=2048)
        target = torch.arange(2048,device='cuda') % 7111 + 1
        for mode in c.MODES:
            net = fresh(mode,'cuda').train(); assert_upstream(net)
            optimizer = torch.optim.Adam(net.parameters(),lr=.001)
            row = dict(memory_mode=mode,status='RUNNING',steps=[])
            record['rows'].append(row); update(c.SMOKE,record)
            torch.cuda.reset_peak_memory_stats()
            selection = None if mode == 'no_memory' else select(data[2],data[0] != 0,mode)
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                captured = []
                def retain(_module,_args,value):
                    value.retain_grad(); captured.append(value)
                handle = net.output_norm.register_forward_hook(retain)
                reader_capture = {}
                def observe(value):
                    reader = value['reader']; selected = value['selection']
                    reader_capture.update(memory_shape=list(reader['values'].shape),
                        selected_causal=bool(((selected['indices'] < torch.arange(50,device='cuda')[None,:,None]) | ~selected['mask']).all()),
                        finite_weights=bool(torch.isfinite(reader['weights']).all()),
                        invalid_weights_zero=bool((reader['weights'][~selected['mask']] == 0).all()),
                        weight_sums=bool(torch.allclose(reader['weights'].sum(-1),selected['mask'].any(-1).float(),atol=1e-6,rtol=1e-5)))
                net.memory_observer = observe
                try:
                    output = net(*data); scores = output @ net.item_embedding.weight.T
                    loss = torch.nn.functional.cross_entropy(scores,target); loss.backward()
                finally:
                    handle.remove(); net.memory_observer = None
                norms = {name:(p.grad.double().norm().item() if p.grad is not None and bool(torch.isfinite(p.grad).all()) else None)
                         for name,p in net.named_parameters()}
                entry = dict(step=step,loss=loss.item() if torch.isfinite(loss) else None,
                             finite_loss=bool(torch.isfinite(loss)),gradient_norms=norms,
                             hook_removed=not net.output_norm._forward_hooks)
                if mode != 'no_memory':
                    raw_gradient = captured[0].grad
                    if raw_gradient is None: raise ValueError('Raw representation gradient missing')
                    last = selection['indices'][:,-1].clamp(min=0)
                    selected_gradient = raw_gradient.gather(1,last[:,:,None].expand(-1,-1,64))
                    entry.update(beta_before=net.beta.item(),lambda_before=torch.tanh(net.beta).item(),
                                 selected_value_gradient_finite=bool(torch.isfinite(selected_gradient).all()),
                                 selected_value_gradient_l2=selected_gradient.double().norm().item(),**reader_capture)
                row['steps'].append(entry); update(c.SMOKE,record)
                if not torch.isfinite(loss) or any(value is None for value in norms.values()): raise ValueError('Nonfinite/missing smoke gradient')
                if mode != 'no_memory' and (reader_capture.get('memory_shape') != [2048,50,4,64] or
                    not all(entry[k] for k in ('selected_causal','finite_weights','invalid_weights_zero','weight_sums','selected_value_gradient_finite'))):
                    raise ValueError('Memory shape/mask/backward smoke failure')
                optimizer.step()
                if mode != 'no_memory':
                    entry.update(beta_after=net.beta.item(),lambda_after=torch.tanh(net.beta).item(),
                                 beta_changed=net.beta.item() != entry['beta_before'])
                update(c.SMOKE,record)
                del captured
            row['beta_update_role'] = 'Recorded diagnostic; required trainability is the informative admission fixture'
            stream = io.BytesIO(); torch.save(net.state_dict(),stream); stream.seek(0)
            saved = torch.load(stream,map_location='cpu',weights_only=True)
            row['roundtrip_passed'] = set(saved) == set(net.state_dict()) and all(torch.equal(value.cpu(),saved[key]) for key,value in net.state_dict().items())
            update(c.SMOKE,record)
            if not row['roundtrip_passed']: raise ValueError('State dict roundtrip failed')
            net.load_state_dict(saved,strict=True)
            row.update(status='PASS',peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),roundtrip='weights_only=True')
            update(c.SMOKE,record)
            del net,optimizer,output,scores,loss,saved,stream,selection
            torch.cuda.empty_cache()
        record['status'] = 'PASS'
    except BaseException as exc:
        record.update(status='BLOCKED_OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'FAIL',
                      error=repr(exc),traceback=traceback.format_exc())
    record['finished_at'] = now(); update(c.SMOKE,record)
    return 0 if record['status'] == 'PASS' else 1


if __name__ == '__main__': raise SystemExit(main())
