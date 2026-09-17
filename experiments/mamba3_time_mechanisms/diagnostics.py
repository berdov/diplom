"""Detached streaming moments and bounded, seeded reservoir quantiles."""

import torch


class TemporalDiagnostics:
    def __init__(self, mode, capacity=8192):
        self.mode, self.capacity = mode, capacity
        self.rng = torch.Generator().manual_seed(2026)
        self.count = 0
        self.sum = torch.zeros(2, 2, dtype=torch.float64)
        self.sq = torch.zeros_like(self.sum)
        self.minimum = torch.full_like(self.sum, float('inf'))
        self.maximum = torch.full_like(self.sum, -float('inf'))
        self.sample = torch.empty(0, 2, 2, dtype=torch.float64)
        self.keys = torch.empty(0)
        self.pairs = torch.zeros(8, dtype=torch.float64)

    @torch.no_grad()
    def update(self, decay, scan, active):
        values = torch.stack((decay.detach()[active], scan.detach()[active]), dim=1).double().cpu()
        if not len(values):
            return
        self.count += len(values)
        self.sum += values.sum(0)
        self.sq += values.square().sum(0)
        self.minimum = torch.minimum(self.minimum, values.amin(0))
        self.maximum = torch.maximum(self.maximum, values.amax(0))
        x, y = values[:, 0].log().flatten(), values[:, 1].log().flatten()
        self.pairs += torch.stack((x.sum(), y.sum(), x.square().sum(), y.square().sum(),
                                   (x*y).sum(), (x-y).abs().sum(), (x>y).double().sum(), (y>x).double().sum()))
        keys = torch.cat((self.keys, torch.rand(len(values), generator=self.rng)))
        sample = torch.cat((self.sample, values))
        keep = keys.topk(min(self.capacity, len(keys))).indices
        self.keys, self.sample = keys[keep], sample[keep]

    def result(self):
        if not self.count:
            return dict(active_history_gaps=0, scales={})
        names = dict(vanilla=(), decay_only=(('decay', 0),), scan_only=(('scan', 1),),
                     shared=(('shared', 0),), separate=(('decay', 0), ('scan', 1)))[self.mode]
        mean = self.sum / self.count
        std = (self.sq / self.count - mean.square()).clamp_min(0).sqrt()
        q = torch.quantile(self.sample, torch.tensor([.1,.25,.5,.75,.9], dtype=torch.float64), dim=0)
        result = dict(active_history_gaps=self.count, padding_and_first_excluded=True,
                      quantiles='seeded uniform reservoir; approximate', sample_size=len(self.sample), scales={})
        for name, index in names:
            result['scales'][name] = dict(mean=mean[index].tolist(), std=std[index].tolist(),
                min=self.minimum[index].tolist(), max=self.maximum[index].tolist(),
                **{label:q[i,index].tolist() for i,label in enumerate(('p10','p25','p50','p75','p90'))})
        if self.mode == 'separate':
            sx, sy, sxx, syy, sxy, sad, gt, lt = self.pairs.tolist()
            n = self.count * 2
            vx, vy = max(sxx/n-(sx/n)**2, 0), max(syy/n-(sy/n)**2, 0)
            corr = (sxy/n-sx*sy/n**2)/(vx*vy)**.5 if vx > 1e-24 and vy > 1e-24 else None
            result['separate'] = dict(mean_abs_log_difference=sad/n,
                log_correlation=None if corr is None else max(-1., min(1., corr)),
                decay_greater_fraction=gt/n, scan_greater_fraction=lt/n)
        return result
