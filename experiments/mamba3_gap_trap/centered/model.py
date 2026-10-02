"""Identical dual backbone and inherited raw-Trap injection point."""
from ..model import GapTrapMamba3Rec as ParentModel
from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
from .modulation import GapTrap


class GapTrapMamba3Rec(ParentModel):
    def __init__(self, config, dataset):
        if config['three_time_mode'] != 'dual':
            raise ValueError('Fixed-reference dual only')
        ThreeTimeMamba3Rec.__init__(self, config, dataset)
        self.gap_trap_mode = config['gap_trap_mode']
        if self.gap_trap_mode not in ('fixed_replay', 'centered_gap_trap'):
            raise ValueError('Unknown centered gap_trap_mode')
        if self.gap_trap_mode == 'centered_gap_trap':
            self.gap_trap = GapTrap()
