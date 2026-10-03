"""Initialization, inherited-path and state tests with a CPU routing fixture.

The fixture only substitutes the unavailable CUDA mixer; it does not stand in
for numerical MIMO admission, which runs independently on the reserved GPU.
"""
import copy
import io
import unittest
from unittest.mock import patch

import torch

from experiments.mamba3_time_memory import config as c
from experiments.mamba3_time_memory.checks import fresh
from experiments.mamba3_time_memory.model import transfer_common
from experiments.mamba3_layer_temporal.checks import cpu_mixer
from experiments.mamba3_three_time.confirmation.state import capture_rng, restore_rng, rng_record


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {}
        cls.rng = {}
        for mode in c.MODES:
            cls.models[mode] = fresh(mode)
            cls.rng[mode] = rng_record()
        cls.historical = fresh('no_memory', historical=True)
        cls.rng['historical'] = rng_record()
        cls.data = (
            torch.tensor([[1, 2, 2, 4, 5, 6, 7, 8], [9, 10, 10, 0, 0, 0, 0, 0]]),
            torch.tensor([8, 3]),
            torch.tensor([[0, 1, 2, 3, 10, 11, 20, 21], [0, 0, 4, 0, 0, 0, 0, 0]], dtype=torch.float64) * 838393)

    def test_counts_and_only_one_new_parameter(self):
        old = dict(self.historical.named_parameters())
        for mode, net in self.models.items():
            params = dict(net.named_parameters())
            self.assertEqual(sum(value.numel() for value in params.values()), c.COUNTS[mode])
            self.assertEqual(set(params) - set(old), {'beta'} if mode != 'no_memory' else set())
            if mode != 'no_memory':
                self.assertEqual(net.beta.shape, torch.Size([]))
                self.assertEqual(net.beta.dtype, torch.float32)
                self.assertEqual(net.beta.item(), 0.)

    def test_constructor_rng_exact_historical_and_no_cuda(self):
        for state in self.rng.values():
            self.assertEqual(state, self.rng['historical'])
        self.assertFalse(torch.cuda.is_initialized())

    def test_common_initialization_and_control_serialization_exact(self):
        old = self.historical.state_dict()
        for net in self.models.values():
            self.assertTrue(all(torch.equal(value, net.state_dict()[key]) for key, value in old.items()))
        def dump(model):
            stream = io.BytesIO()
            torch.save(model.state_dict(), stream)
            return stream.getvalue()
        self.assertEqual(dump(self.historical), dump(self.models['no_memory']))

    def test_historical_state_transfer_without_memory_attribute(self):
        self.assertFalse(hasattr(self.historical, 'memory_mode'))
        for mode in c.MODES:
            net = copy.deepcopy(self.models[mode])
            result = transfer_common(self.historical, net)
            self.assertTrue(result['all_keys_shapes_dtypes_values_equal'])
            self.assertFalse(result['source_has_beta'])
            self.assertTrue(all(torch.equal(value, net.state_dict()[key]) for key, value in self.historical.state_dict().items()))

    def test_transfer_rejects_unrelated_state_mismatch(self):
        source = copy.deepcopy(self.historical)
        source.register_buffer('unexpected', torch.ones(1))
        with self.assertRaises(ValueError):
            transfer_common(source, self.models['index_memory'])
        source = copy.deepcopy(self.historical)
        source.output_norm.weight = torch.nn.Parameter(source.output_norm.weight.double())
        with self.assertRaises(ValueError):
            transfer_common(source, self.models['time_memory'])

    def test_beta0_forward_common_gradients_and_dropout_rng(self):
        reference = copy.deepcopy(self.historical).train()
        state = capture_rng()
        with patch('experiments.mamba3_three_time.model.three_time_forward', cpu_mixer):
            expected = reference.encode_sequence(*self.data)
            expected.square().sum().backward()
            expected_rng = rng_record()
            gradients = {key: value.grad for key, value in reference.named_parameters()}
            for mode in c.MODES:
                net = copy.deepcopy(self.models[mode]).train()
                restore_rng(state)
                actual = net.encode_sequence(*self.data)
                actual.square().sum().backward()
                self.assertTrue(torch.equal(actual, expected), mode)
                self.assertEqual(rng_record(), expected_rng)
                for key, value in net.named_parameters():
                    if key == 'beta':
                        self.assertIsNotNone(value.grad)
                        self.assertTrue(torch.isfinite(value.grad))
                    elif gradients[key] is None:
                        self.assertIsNone(value.grad)
                    else:
                        self.assertTrue(torch.equal(value.grad, gradients[key]), (mode, key))

    def test_one_encoder_pass_original_archive_and_observer_detachment(self):
        net = copy.deepcopy(self.models['time_memory'])
        net.beta.data.fill_(.3)
        observed, encoded = [], []
        net.memory_observer = observed.append
        handle = net.output_norm.register_forward_hook(lambda module, inputs, output: encoded.append(output))
        try:
            with patch('experiments.mamba3_three_time.model.three_time_forward', cpu_mixer):
                output = net.encode_sequence(*self.data)
        finally:
            handle.remove()
            net.memory_observer = None
        self.assertEqual(len(encoded), 1)
        self.assertEqual(len(observed), 1)
        payload = observed[0]
        self.assertTrue(torch.equal(payload['H'], encoded[0]))
        self.assertEqual(payload['H'].data_ptr(), encoded[0].data_ptr())
        self.assertFalse(payload['H'].requires_grad)
        self.assertFalse(payload['reader']['values'].requires_grad)
        self.assertTrue(output.requires_grad)
        self.assertEqual(payload['reader']['memory_shape'], [2, 8, 4, 64])
        self.assertFalse(net.output_norm._forward_hooks)

    def test_last_valid_residual_uses_inherited_scorer_path(self):
        net = copy.deepcopy(self.models['index_memory'])
        net.beta.data.fill_(.2)
        with patch('experiments.mamba3_three_time.model.three_time_forward', cpu_mixer):
            sequence = net.encode_sequence(*self.data)
            actual = net(*self.data)
        self.assertTrue(torch.equal(actual, sequence[torch.arange(2), self.data[1] - 1]))

    def test_consecutive_forward_no_persistent_memory(self):
        net = copy.deepcopy(self.models['time_memory'])
        net.beta.data.fill_(.3)
        state_before = set(net.state_dict())
        modified = (self.data[0] + (self.data[0] != 0) * 20, self.data[1], self.data[2] * 100)
        with patch('experiments.mamba3_three_time.model.three_time_forward', cpu_mixer), torch.no_grad():
            first = net.encode_sequence(*self.data)
            net.encode_sequence(*modified)
            last = net.encode_sequence(*self.data)
        self.assertTrue(torch.equal(first, last))
        self.assertEqual(state_before, set(net.state_dict()))
        self.assertFalse(any(torch.is_tensor(value) for value in vars(net).values()))
        self.assertIsNone(net.memory_observer)

    def test_full_model_suffix_padding_and_cross_user_interventions(self):
        for mode in ('index_memory', 'time_memory'):
            net = copy.deepcopy(self.models[mode])
            net.beta.data.fill_(.4)
            items, lengths, times = (value.clone() for value in self.data)
            items[0, 5:] += 20
            times[0, 5:] = torch.tensor([1e9, 2e9, 3e9], dtype=torch.float64)
            times[1, 3:] = float('nan')
            with patch('experiments.mamba3_three_time.model.three_time_forward', cpu_mixer), torch.no_grad():
                before = net.encode_sequence(*self.data)
                after = net.encode_sequence(items, lengths, times)
            self.assertTrue(torch.equal(before[0, :5], after[0, :5]))
            self.assertTrue(torch.equal(before[1], after[1]))

    def test_state_dict_weights_only_roundtrip(self):
        for mode in c.MODES:
            net = copy.deepcopy(self.models[mode])
            if mode != 'no_memory':
                net.beta.data.fill_(-.37)
            stream = io.BytesIO()
            torch.save(net.state_dict(), stream)
            stream.seek(0)
            saved = torch.load(stream, map_location='cpu', weights_only=True)
            clone = copy.deepcopy(self.models[mode])
            clone.load_state_dict(saved, strict=True)
            self.assertTrue(all(torch.equal(value, clone.state_dict()[key]) for key, value in net.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
