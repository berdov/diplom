"""CPU construction and strict state transfer; native execution has its own gate."""

import unittest

import torch
from torch import nn

from experiments.mamba3_absolute_phase.phase import PeriodicPhase
from experiments.mamba3_absolute_phase.model import transfer_common


class Container(nn.Module):
    def __init__(self, modified=False, width=3, extra=False):
        super().__init__()
        self.common = nn.Linear(width, 2)
        self.register_buffer('reference', torch.tensor(838393., dtype=torch.float64))
        self.phase_adapter = PeriodicPhase(32, 'absolute_phase') if modified else None
        if extra:
            self.unexpected = nn.Parameter(torch.ones(1))


class MappingTests(unittest.TestCase):
    def test_actual_construction_counts_common_state_rng_and_cpu(self):
        from experiments.mamba3_absolute_phase.checks import fresh
        from experiments.mamba3_three_time.confirmation.state import rng_record
        self.assertFalse(torch.cuda.is_initialized())
        original = fresh('baseline_dual', device='cpu', historical=True)
        initial_rng = rng_record()
        source = original.state_dict()
        for mode, count in [('baseline_dual', 715020), ('relative_phase', 715148),
                            ('absolute_phase', 715148)]:
            model = fresh(mode, device='cpu')
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            self.assertEqual({layer.mixer.num_rope_angles for layer in model.layers}, {32})
            self.assertEqual(rng_record(), initial_rng)
            self.assertFalse(torch.cuda.is_initialized())
            self.assertEqual(set(model.state_dict())-set(source),
                             set() if mode == 'baseline_dual' else {'phase_adapter.W'})
            for key, value in source.items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
            if mode == 'baseline_dual':
                self.assertIsNone(model.phase_adapter)
            else:
                self.assertEqual(model.phase_adapter.W.dtype, torch.float32)

    def test_only_new_weight_excluded(self):
        source, target = Container(), Container(modified=True)
        with torch.no_grad():
            target.phase_adapter.W.fill_(.37)
        keys = transfer_common(source, target)
        self.assertEqual(keys, ['common.bias', 'common.weight', 'reference'])
        for key in keys:
            self.assertTrue(torch.equal(source.state_dict()[key], target.state_dict()[key]))
        self.assertTrue(torch.equal(target.phase_adapter.W, torch.full((32, 4), .37)))

    def test_extra_key_rejected(self):
        with self.assertRaisesRegex(ValueError, 'keys differ'):
            transfer_common(Container(), Container(modified=True, extra=True))

    def test_shape_drift_rejected(self):
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            transfer_common(Container(), Container(width=4))

    def test_dtype_drift_rejected(self):
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            transfer_common(Container(), Container().double())


if __name__ == '__main__':
    unittest.main()
