"""Real saved-file failure paths; no model, checkpoint loading or GPU work."""
import copy
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from experiments.mamba3_absolute_phase import config as c, pipeline, report
from experiments.mamba3_mimo_time.records import create, read, sha


@contextmanager
def scratch():
    plan = (c.HERE / 'study_plan.json').read_bytes()
    with tempfile.TemporaryDirectory() as directory, ExitStack() as patches:
        here = Path(directory) / 'study'
        here.mkdir()
        (here / 'study_plan.json').write_bytes(plan)
        values = dict(ROOT=Path(directory), HERE=here,
                      LOGS=here / 'slurm_logs/attempt_001', RUNS=here / 'runs/attempt_001',
                      SUMMARY=here / 'runs/attempt_001/pilot_summary.json',
                      PIPELINE=here / 'slurm_logs/attempt_001/pipeline_status.json')
        for key, value in values.items():
            patches.enter_context(patch.object(c, key, value))
        yield


def owner():
    return dict(execution_attempt='001', study_phase='pilot', execution_commit='a' * 40,
                source_hash='b' * 64, job_id='fixture', TEST='NOT_RUN', test_evaluation_count=0)


def raw_record(variant, **changes):
    return dict(owner(), phase_mode=variant, mode='dual', seed=2026,
                run_id=c.paths(variant)['run_id'], **changes)


class FailureRecordTests(unittest.TestCase):
    def assert_unknown(self, summary, variant='relative_phase'):
        self.assertEqual(summary['status'], 'INCOMPLETE')
        self.assertEqual(summary['unknown_scientific_starts'], 1)
        self.assertEqual(pipeline.fit_counters(), dict(scientific_fits_started=0,
                         scientific_fits_completed=0, unknown_scientific_starts=1))
        row = next(x for x in summary['rows'] if x['variant'] == variant)
        self.assertEqual(row['status'], 'UNKNOWN')
        self.assertIsNone(row['scientific_fit_started'])
        self.assertIsNone(row['actual_epochs'])
        self.assertIsNone(row['best_valid_metrics'])
        self.assertTrue(row['validation_error'])
        self.assertTrue(all(x['delta'] is None for x in summary['contrasts']))

    def test_owner_without_result_remains_unknown_after_real_report_write(self):
        for artifact in ('lock', 'checkpoint', 'metadata', 'progress'):
            with self.subTest(artifact=artifact), scratch():
                paths = c.paths('relative_phase')
                path = paths['runtime'] / 'progress.json' if artifact == 'progress' else paths[artifact]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'opaque owner evidence; never deserialize')
                before = sha(path)
                self.assertEqual(pipeline.fit_counters()['unknown_scientific_starts'], 1)
                summary = report.write(owner(), 'synthetic missing owner result')
                self.assert_unknown(summary)
                self.assertFalse(paths['result'].exists())
                self.assertEqual(sha(path), before)

    def test_damaged_or_nonobject_result_remains_byte_identical(self):
        for content in (b'{', b'null', b'[]', b'"text"', b'17', b'false', b'\xff'):
            with self.subTest(content=content), scratch():
                path = c.paths('relative_phase')['result']
                path.parent.mkdir(parents=True)
                path.write_bytes(content)
                self.assertEqual(pipeline.fit_counters()['unknown_scientific_starts'], 1)
                self.assert_unknown(report.write(owner(), 'synthetic damaged result'))
                self.assertEqual(path.read_bytes(), content)

    def test_real_pipeline_finally_preserves_unknown_in_terminal(self):
        with scratch(), patch.object(pipeline, 'identity', return_value=owner()), patch.object(pipeline.signal, 'signal'):
            paths = c.paths('relative_phase')
            create(paths['lock'], raw_record('relative_phase', status='RUNNING', scientific_fit_started=False))
            # No CPU proofs: the actual pipeline fails before admission, then
            # runs its real report/final counter/terminal preservation path.
            self.assertEqual(pipeline.run(), 1)
            self.assert_unknown(read(c.SUMMARY))
            status = read(c.PIPELINE)
            terminal = read(c.RUNS / 'terminal_metadata.json')
            for value in (status, terminal):
                self.assertEqual(value['unknown_scientific_starts'], 1)
                self.assertEqual(value['scientific_fits_started'], 0)
                self.assertEqual(value['scientific_fits_completed'], 0)
                self.assertEqual(value['summary_status'], 'INCOMPLETE')
                self.assertNotIn('report_traceback', value)
            self.assertEqual(status['stages'], [])
            self.assertEqual(terminal['checkpoints'], {})
            self.assertFalse(paths['result'].exists())

    def test_missing_or_nonbool_start_flag_does_not_become_false(self):
        for flag in ('missing', None, 0, 1, 'false', 'true', []):
            with self.subTest(flag=flag), scratch():
                record = raw_record('relative_phase', status='FAIL', actual_epochs=0, history=[])
                if flag != 'missing':
                    record['scientific_fit_started'] = flag
                path = c.paths('relative_phase')['result']
                create(path, record)
                before = path.read_bytes()
                self.assert_unknown(report.write(owner(), 'synthetic invalid start flag'))
                self.assertEqual(path.read_bytes(), before)

    def test_dangling_result_symlink_is_not_replaced(self):
        with scratch():
            paths = c.paths('relative_phase')
            paths['result'].parent.mkdir(parents=True)
            target = paths['result'].with_name('absent.json')
            paths['result'].symlink_to(target)
            self.assert_unknown(report.write(owner(), 'synthetic dangling result link'))
            self.assertTrue(paths['result'].is_symlink())
            self.assertEqual(paths['result'].readlink(), target)
            self.assertFalse(target.exists())

    def test_no_artifacts_stays_known_not_run(self):
        with scratch():
            summary = report.write(owner(), 'synthetic failure before child launch')
            self.assertEqual(summary['unknown_scientific_starts'], 0)
            self.assertEqual(pipeline.fit_counters(), dict(scientific_fits_started=0,
                             scientific_fits_completed=0, unknown_scientific_starts=0))
            for variant in c.MODES:
                self.assertFalse(read(c.paths(variant)['result'])['scientific_fit_started'])
                self.assertEqual(read(c.paths(variant)['result'])['status'], 'NOT_RUN')

    def test_genuine_setup_failure_with_lock_is_known_not_started(self):
        with scratch():
            paths = c.paths('relative_phase')
            record = raw_record('relative_phase', status='FAIL', stage='SETUP',
                                scientific_fit_started=False, actual_epochs=0, history=[])
            create(paths['lock'], record)
            create(paths['result'], record)
            before = paths['result'].read_bytes()
            summary = report.write(owner(), 'synthetic setup failure')
            self.assertEqual(summary['unknown_scientific_starts'], 0)
            self.assertEqual(pipeline.fit_counters()['unknown_scientific_starts'], 0)
            self.assertEqual(paths['result'].read_bytes(), before)

    def test_not_run_placeholder_cannot_hide_existing_owner(self):
        with scratch():
            paths = c.paths('relative_phase')
            create(paths['lock'], {})
            create(paths['result'], raw_record('relative_phase', status='NOT_RUN',
                   scientific_fit_started=False, actual_epochs=0, history=[]))
            before = paths['result'].read_bytes()
            self.assert_unknown(report.write(owner(), 'synthetic legacy placeholder'))
            self.assertEqual(paths['result'].read_bytes(), before)

    def test_false_start_with_training_evidence_is_unknown(self):
        for artifact in ('checkpoint', 'metadata', 'PASS', 'epochs', 'history'):
            with self.subTest(artifact=artifact), scratch():
                paths = c.paths('relative_phase')
                create(paths['result'], raw_record('relative_phase', status='PASS' if artifact == 'PASS' else 'FAIL',
                       scientific_fit_started=False, actual_epochs=1 if artifact == 'epochs' else 0,
                       history=[{'epoch': 0}] if artifact == 'history' else []))
                if artifact in ('checkpoint', 'metadata'):
                    create(paths[artifact], {})
                self.assert_unknown(report.write(owner(), 'synthetic contradictory evidence'))

    def test_partial_fit_stays_counted_and_raw_is_unchanged(self):
        with scratch():
            paths = c.paths('relative_phase')
            record = raw_record('relative_phase', status='INCOMPLETE', scientific_fit_started=True,
                                actual_epochs=1, history=[{'epoch': 0}], error='synthetic interruption')
            create(paths['result'], record)
            before = paths['result'].read_bytes()
            summary = report.write(owner(), 'synthetic interrupted fit')
            self.assertEqual(summary['scientific_fits_started'], 1)
            self.assertEqual(summary['unknown_scientific_starts'], 0)
            self.assertEqual(pipeline.fit_counters(), dict(scientific_fits_started=1,
                             scientific_fits_completed=0, unknown_scientific_starts=0))
            self.assertEqual(paths['result'].read_bytes(), before)

    def test_foreign_owner_is_still_rejected_without_rewriting(self):
        with scratch():
            path = c.paths('baseline_dual')['result']
            record = raw_record('baseline_dual', status='FAIL', scientific_fit_started=False)
            record['job_id'] = 'foreign'
            create(path, record)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'Foreign run record'):
                report.write(owner())
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(c.SUMMARY.exists())

    def test_valid_control_summary_keeps_exact_scientific_values(self):
        # Real immutable historical JSON, adapted only as the new no-memory
        # record; no checkpoint is opened and no model operation is executed.
        historical = read(c.PILOT)
        record = copy.deepcopy(historical)
        record.update(phase_mode='baseline_dual', run_id=c.paths('baseline_dual')['run_id'],
                      initial_common_calibrator_hashes=record['initial_calibrator_hashes'], initial_phase_parameters={})
        for key in ('config', 'effective_config'):
            record[key]['phase_mode'] = 'baseline_dual'
        summary = report.summarize({('baseline_dual',2026): record})
        row = summary['rows'][0]
        self.assertEqual(row['status'], 'PASS')
        self.assertEqual(summary['unknown_scientific_starts'], 0)
        for key in ('best_valid_metrics', 'best_epoch', 'actual_epochs', 'first27_complete',
                    'first27_best_ndcg10', 'train_seconds', 'valid_seconds', 'peak_gpu_allocated_bytes',
                    'peak_gpu_reserved_bytes', 'checkpoint_sha256', 'best_diagnostics'):
            self.assertEqual(row[key], record[key])
        self.assertEqual(record['history'], historical['history'])


if __name__ == '__main__':
    unittest.main()
