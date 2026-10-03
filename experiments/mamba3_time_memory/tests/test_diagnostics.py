"""Coverage and VALID evidence are bounded, input-only, and RNG-neutral."""
import json
import random
import unittest

import torch

from experiments.mamba3_time_memory.coverage import (
    MAX_EXAMPLES, UNINFORMATIVE, audit_histories, fixed_indices, tensor_fingerprint,
)
from experiments.mamba3_time_memory.diagnostics import (
    AddressDiagnostics, MemoryDiagnostics, QueryReservoir,
)
from experiments.mamba3_time_memory.memory import select, read_memory

R0 = 838393


def fixture():
    lengths = torch.tensor([1, 8, 8])
    valid = torch.arange(8)[None] < lengths[:, None]
    items = torch.where(valid, torch.arange(1, 9)[None].expand(3, -1), 0)
    timestamps = torch.arange(8, dtype=torch.float64)[None].expand(3, -1).clone() * R0
    timestamps[2] = torch.tensor([0, 1, 2, 3, 4, 5, 6, 20], dtype=torch.float64) * R0
    timestamps[0, 1:] = float('nan')
    return items, lengths, timestamps, valid


class CoverageTests(unittest.TestCase):
    def test_fixed_indices_integers_endpoints_and_bound(self):
        self.assertEqual(fixed_indices(1), [0])
        self.assertEqual(fixed_indices(4), [0, 1, 2, 3])
        self.assertEqual(fixed_indices(12, 5), [0, 2, 5, 8, 11])
        indices = fixed_indices(1062567)
        self.assertEqual(len(indices), MAX_EXAMPLES)
        self.assertEqual((indices[0], indices[-1]), (0, 1062566))
        self.assertEqual(len(set(indices)), len(indices))
        for args in ((0,), (1.5,), (5, 10001), (5, 0)):
            with self.assertRaises(ValueError):
                fixed_indices(*args)

    def test_coverage_only_last_query_and_different_addresses(self):
        items, lengths, timestamps, _ = fixture()
        result = audit_histories(items, lengths, timestamps, [4, 19, 99], batch_size=2)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['examples'], 3)
        self.assertEqual(result['history_length_histogram'], {'1': 1, '8': 2})
        self.assertEqual(result['valid_slot_count_histogram'], {'0': 1, '1': 0, '2': 0, '3': 0, '4': 2})
        self.assertEqual(result['index_memory']['selected_event_lags']['count'], 8)
        self.assertEqual(result['time_memory']['selected_event_lags']['count'], 8)
        self.assertEqual(result['empty_memory_fraction'], 1 / 3)
        self.assertEqual(result['selected_sets_equal_count'], 2)
        self.assertEqual(result['temporal_span_R0']['max'], 20)
        self.assertEqual({r['input_index'] for r in result['examples_reservoir']}, {4, 19, 99})
        json.dumps(result, allow_nan=False)

    def test_identical_sets_are_uninformative_no_threshold_search(self):
        items, lengths, timestamps, _ = fixture()
        result = audit_histories(items[:2], lengths[:2], timestamps[:2], [0, 1])
        self.assertEqual(result['status'], UNINFORMATIVE)
        self.assertEqual(result['selected_sets_equal_fraction'], 1.0)

    def test_selected_source_hashes_bind_exact_timestamps(self):
        items, lengths, timestamps, _ = fixture()
        first = audit_histories(items, lengths, timestamps, [0, 1, 2])
        shifted = timestamps.clone()
        shifted[2] += 1234
        second = audit_histories(items, lengths, shifted, [0, 1, 2])
        self.assertNotEqual(first['selected_records_sha256'], second['selected_records_sha256'])
        self.assertEqual(first['selected_input_hashes']['item_history'], second['selected_input_hashes']['item_history'])
        self.assertEqual(first['selected_sets_equal_fraction'], second['selected_sets_equal_fraction'])
        self.assertNotEqual(tensor_fingerprint(lengths), tensor_fingerprint(lengths.int()))

    def test_padding_and_precision_fail_closed(self):
        items, lengths, timestamps, _ = fixture()
        with self.assertRaises(ValueError):
            audit_histories(items, lengths, timestamps.float(), [0, 1, 2])
        items[1, 3] = 0
        with self.assertRaises(ValueError):
            audit_histories(items, lengths, timestamps, [0, 1, 2])


class DiagnosticsTests(unittest.TestCase):
    def test_bounded_reservoir_chunk_invariant_without_rng_draws(self):
        py_before, torch_before = random.getstate(), torch.get_rng_state().clone()
        one, two = QueryReservoir(7), QueryReservoir(7)
        for i in range(1000):
            one.add(i, {'i': i})
        for start in range(0, 1000, 13):
            for i in range(start, min(1000, start + 13)):
                two.add(i, {'i': i})
        self.assertEqual(one.rows(), two.rows())
        self.assertEqual(len(one.rows()), 7)
        self.assertEqual(py_before, random.getstate())
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))

    def test_valid_callback_negative_gate_and_no_retained_representations(self):
        items, lengths, timestamps, valid = fixture()
        hidden = torch.arange(3 * 8 * 4, dtype=torch.float32).reshape(3, 8, 4) / 100
        beta = torch.tensor(-.2)
        selection = select(timestamps, valid, 'time_memory')
        output, reader = read_memory(hidden, selection, beta)
        payload = dict(H=hidden, output=output, reader=reader, selection=selection,
                       beta=beta, **{'lambda': beta.tanh()}, valid=valid,
                       timestamps=timestamps, mode='time_memory')
        py_before, torch_before = random.getstate(), torch.get_rng_state().clone()
        collector = MemoryDiagnostics('time_memory')
        collector.observe(payload)
        result = collector.result()
        self.assertEqual(result['status'], 'MEASURED')
        self.assertEqual(result['examples'], 3)
        self.assertEqual(result['reader_weights']['count'], 8)
        self.assertTrue(result['negative_gate'])
        self.assertEqual(result['beta'], beta.item())
        self.assertEqual(result['reader_bank_shapes'], [[3, 8, 4, 4]])
        self.assertEqual(result['max_reader_bank_elements'], 384)
        self.assertEqual(result['empty_memory_fraction'], 1 / 3)
        self.assertTrue(all('H' not in row for row in result['examples_reservoir']))
        self.assertEqual(py_before, random.getstate())
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        json.dumps(result, allow_nan=False)
        payload['beta'] = torch.tensor(.1)
        with self.assertRaises(ValueError):
            collector.observe(payload)

    def test_counts_and_reservoir_stable_across_last_batches(self):
        items, lengths, timestamps, valid = fixture()
        full = AddressDiagnostics('fixture')
        chunks = AddressDiagnostics('fixture')
        full.update(timestamps, valid)
        chunks.update(timestamps[:2], valid[:2])
        chunks.update(timestamps[2:], valid[2:])
        self.assertEqual(full.result(), chunks.result())


if __name__ == '__main__':
    unittest.main()
