"""Detached statistics collected on existing VALID forwards, not extra evaluation."""
import torch
from experiments.mamba3_time_mechanisms.diagnostics import TemporalDiagnostics


class Diagnostics:
    def __init__(self, mode):
        self.mode = mode
        self.dw = TemporalDiagnostics('separate', capacity=2048)
        self.wp = TemporalDiagnostics('separate', capacity=2048)
        self.count = 0
        self.bounds = torch.zeros(3, 2, 2, dtype=torch.float64)
        self.log_difference = torch.zeros(2, dtype=torch.float64)
        self.equal_write_phase = True

    @torch.no_grad()
    def hook(self, _module, args, output):
        active = args[1]
        decay, write, phase = output
        self.dw.update(decay, write, active)
        self.wp.update(write, phase, active)
        values = torch.stack([x.detach()[active] for x in output], dim=1).double().cpu()
        if not len(values):
            return
        self.count += len(values)
        self.bounds[:, :, 0] += (values < .51).double().sum(0)
        self.bounds[:, :, 1] += (values > 1.99).double().sum(0)
        self.log_difference += (values[:, 1].log() - values[:, 2].log()).abs().sum(0)
        self.equal_write_phase &= torch.equal(write.detach(), phase.detach())

    def result(self):
        dw, wp = self.dw.result(), self.wp.result()
        if not self.count:
            return dict(status='NO_ACTIVE_GAPS')
        return dict(status='MEASURED', active_history_gaps=self.count, temporal_heads=2,
            shared_across_layers=True, scales=dict(decay=dw['scales']['decay'],
                write=dw['scales']['scan'], phase=wp['scales']['scan']),
            write_equals_phase=self.equal_write_phase,
            mean_abs_write_phase_log_difference_per_head=(self.log_difference / self.count).tolist(),
            write_phase_log_comparison=wp['separate'], near_bound_fractions=self.bounds.div(self.count).tolist(),
            near_bound_axes=['decay/write/phase', 'temporal_head', 'below .51/above 1.99'],
            sampling='separate seeded CPU reservoirs; approximate quantiles; no new forward')
