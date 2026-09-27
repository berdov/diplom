"""Fixed full scientific shapes, synthetic CE/update only; no batch fallback."""
import io
import traceback
import torch
from . import config as c
from .records import create, update, now
from .provenance import identity, require_gate, runtime, assert_upstream
from .admission import fresh, histories


def main():
    base = identity()
    record = dict(**base, status='RUNNING', admission_sha256=require_gate(base), scientific_fits=0,
                  batch=2048, history_length=50, kernel_length=56, steps_per_mode=2, rows=[])
    create(c.SMOKE, record)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        record['runtime'] = runtime(True)
        data = histories(50,batch=2048)
        target = torch.arange(2048,device='cuda')%7111+1
        for mode in c.MODES:
            net = fresh(mode).train()
            optimizer = torch.optim.Adam(net.parameters(),lr=.001)
            row = dict(mode=mode,status='RUNNING',steps=[])
            record['rows'].append(row); update(c.SMOKE,record)
            torch.cuda.reset_peak_memory_stats()
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                assert_upstream(net)
                output = net(*data)
                scores = output @ net.item_embedding.weight.T
                loss = torch.nn.functional.cross_entropy(scores,target)
                loss.backward()
                norms = {n:(p.grad.double().norm().item() if p.grad is not None and bool(torch.isfinite(p.grad).all()) else None)
                         for n,p in net.named_parameters()}
                row['steps'].append(dict(step=step,loss=loss.item() if bool(torch.isfinite(loss)) else None,
                                         finite_loss=bool(torch.isfinite(loss)),gradient_norms=norms))
                update(c.SMOKE,record)
                if not torch.isfinite(loss) or any(v is None for v in norms.values()):
                    raise ValueError('Nonfinite/missing smoke gradients')
                optimizer.step()
            if any(not any(r['gradient_norms'][n]>0 for r in row['steps'][1:])
                   for n,_ in net.named_parameters() if n.startswith('times.')):
                raise ValueError('Calibrator not learnable after zero-init first step')
            stream=io.BytesIO(); torch.save(net.state_dict(),stream); stream.seek(0)
            saved=torch.load(stream,map_location='cpu',weights_only=True)
            if any(not torch.equal(v.cpu(),saved[k]) for k,v in net.state_dict().items()):
                raise ValueError('Pure state_dict roundtrip')
            row.update(status='PASS',peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),roundtrip='weights_only=True')
            update(c.SMOKE,record)
            del net,optimizer,output,scores,loss,saved,stream
            torch.cuda.empty_cache()
        record['status']='PASS'
    except BaseException as exc:
        record.update(status='BLOCKED_OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'FAIL',
                      error=repr(exc),traceback=traceback.format_exc())
    record['finished_at']=now(); update(c.SMOKE,record)
    return 0 if record['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
