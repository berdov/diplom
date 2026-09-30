"""Only replace the calibrators; keep the inherited dual forward and kernels."""
from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
from .calibrator import LearnedReference


class HeadTimescaleMamba3Rec(ThreeTimeMamba3Rec):
    def __init__(self, config, dataset):
        if config['three_time_mode'] != 'dual':
            raise ValueError('This study keeps dual')
        super().__init__(config, dataset)
        self.time_scale_mode = config['time_scale_mode']
        if self.time_scale_mode not in ('fixed', 'shared_tau', 'head_tau'):
            raise ValueError('Unknown time_scale_mode')
        if self.time_scale_mode != 'fixed':
            for name in ('decay', 'scan'):
                self.times.calibrators[name] = LearnedReference(self.times.calibrators[name], self.time_scale_mode)
