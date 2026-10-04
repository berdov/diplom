"""Actual reservation ledger and saved pilot evidence, without scheduler calls."""
import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from experiments.mamba3_absolute_phase import config as c, provenance, submit
from experiments.mamba3_mimo_time.records import sha


class FileFixture(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.here = self.root / 'experiments/mamba3_absolute_phase'
        self.here.mkdir(parents=True)
        for name, value in dict(ROOT=self.root, HERE=self.here, STAGE='pilot',
                                EXECUTION_ATTEMPT='001', SEEDS=(2026,)).items():
            self.stack.enter_context(patch.object(c, name, value))
        # Fail if a unit test accidentally reaches a scheduler/subprocess path.
        self.stack.enter_context(patch.object(submit.subprocess, 'run', side_effect=AssertionError('No jobs in unit tests')))

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False) + '\n')
        return path

    def reserve(self, stage, attempt, seconds):
        value = dict(study_id=c.STUDY, study_phase=stage, execution_attempt=attempt,
                     requested_seconds=seconds, jobs_requested=1, status='RESERVED',
                     scientific_fits_before_submit=0,
                     max_scientific_fits=3 if stage == 'pilot' else 12,
                     token='a' * 32)
        return self.write(self.here / 'slurm_logs' / stage / ('attempt_' + attempt) / 'reservation.json', value)


class BudgetTests(FileFixture):
    def test_real_glob_counts_both_stages_and_allows_exact_18h(self):
        self.reserve('pilot', '001', 21600)
        self.reserve('confirmation', '001', 28800)
        # Unrelated runtime logs must not count as submitted allocations.
        self.write(self.here / 'runtime/reservation.json', dict(requested_seconds=999999))
        with patch.object(c, 'STAGE', 'confirmation'), patch.object(c, 'EXECUTION_ATTEMPT', '002'):
            result = submit.budget_review()
        self.assertEqual(result, dict(previous_submissions=2, requested_seconds_before=50400,
                                      requested_seconds_after=64800))

    def test_fourth_reservation_rejected_at_global_submit_cap(self):
        self.reserve('pilot', '001', 21600)
        self.reserve('pilot', '002', 14400)
        self.reserve('confirmation', '001', 28800)
        with patch.object(c, 'STAGE', 'confirmation'), patch.object(c, 'EXECUTION_ATTEMPT', '002'):
            with self.assertRaises(ValueError):
                submit.budget_review()

    def test_walltime_above_18h_rejected(self):
        self.reserve('pilot', '001', 21600)
        self.reserve('confirmation', '001', 28800)
        with patch.object(c, 'STAGE', 'confirmation'), patch.object(c, 'EXECUTION_ATTEMPT', '002'), \
                patch.object(c, 'allocation_seconds', return_value=14401):
            with self.assertRaises(ValueError):
                submit.budget_review()

    def test_one_retry_is_global_not_per_phase(self):
        self.reserve('pilot', '002', 14400)
        with patch.object(c, 'STAGE', 'confirmation'), patch.object(c, 'EXECUTION_ATTEMPT', '002'):
            with self.assertRaises(ValueError):
                submit.budget_review()

    def test_same_phase_attempt_cannot_reserve_twice(self):
        self.reserve('pilot', '001', 21600)
        with self.assertRaises(ValueError):
            submit.budget_review()

    def test_invalid_prior_walltime_cannot_reduce_or_corrupt_budget(self):
        path = self.reserve('pilot', '001', 21600)
        original = json.loads(path.read_text())
        with patch.object(c, 'STAGE', 'confirmation'):
            for seconds in (-1, 0, True, 21600.5, 21601):
                with self.subTest(seconds=seconds):
                    self.write(path, dict(original, requested_seconds=seconds))
                    with self.assertRaises(ValueError):
                        submit.budget_review()

    def test_ledger_phase_attempt_must_match_canonical_directory(self):
        path = self.reserve('pilot', '001', 21600)
        value = json.loads(path.read_text())
        self.write(path, dict(value, study_phase='confirmation', requested_seconds=28800,
                              max_scientific_fits=12))
        with patch.object(c, 'STAGE', 'confirmation'), patch.object(c, 'EXECUTION_ATTEMPT', '002'):
            with self.assertRaises(ValueError):
                submit.budget_review()


class RetryTests(FileFixture):
    def prepare_retry(self):
        self.stack.enter_context(patch.object(c, 'EXECUTION_ATTEMPT', '002'))
        self.write(self.here / 'slurm_logs/pilot/attempt_001/submission.json', dict(job_id='12345'))
        preserved = self.write(self.here / 'evidence/parent/preservation_manifest.json', dict(files=[]))
        review = dict(status='APPROVED_PRE_FIT_TECHNICAL_RETRY', scientific_fits_started=0,
                      unknown_scientific_starts=0, method_unchanged=True, tolerances_unchanged=True,
                      regression_passed=True, infrastructure_error='reproduced wrapper error',
                      old_job_terminal=True, old_job_id='12345',
                      scheduler=dict(JobIDRaw='12345', State='FAILED'),
                      preservation_manifest_path=str(preserved.relative_to(self.root)),
                      preservation_manifest_sha256=sha(preserved))
        path = self.write(self.here / 'runtime/retry_review_pilot.json', review)
        for task in c.tasks():
            self.write(self.here / 'runs/pilot/attempt_001' / (task['run_id'] + '.json'),
                       dict(status='NOT_RUN', scientific_fit_started=False, actual_epochs=0, history=[]))
        return path, review

    def test_review_returns_review_hash_not_last_scientific_json_hash(self):
        path, _ = self.prepare_retry()
        self.assertEqual(submit.retry_review(), sha(path))

    def test_started_scientific_fit_never_retried(self):
        self.prepare_retry()
        task = c.tasks()[0]
        self.write(self.here / 'runs/pilot/attempt_001' / (task['run_id'] + '.json'),
                   dict(status='FAIL', scientific_fit_started=True, actual_epochs=0, history=[]))
        with self.assertRaises(ValueError):
            submit.retry_review()

    def test_unknown_owner_without_result_never_retried(self):
        self.prepare_retry()
        task = c.tasks()[0]
        result = self.here / 'runs/pilot/attempt_001' / (task['run_id'] + '.json')
        result.unlink()  # Synthetic fixture only.
        self.write(self.here / 'slurm_logs/pilot/attempt_001' / task['run_id'] / 'progress.json', {})
        with self.assertRaises(ValueError):
            submit.retry_review()

    def test_false_start_flag_with_checkpoint_evidence_never_retried(self):
        self.prepare_retry()
        task = c.tasks()[0]
        self.write(self.here / 'slurm_logs/pilot/attempt_001' / task['run_id'] /
                   'checkpoints/best_metadata.json', dict(epoch=0))
        with self.assertRaises(ValueError):
            submit.retry_review()

    def test_initial_phase_cannot_restart_after_retry_reservation(self):
        self.reserve('pilot', '002', 14400)
        with self.assertRaises(ValueError):
            submit.retry_review()

    def test_progress_must_agree_with_pre_fit_result(self):
        path, _ = self.prepare_retry()
        task = c.tasks()[0]
        self.write(self.here / 'runs/pilot/attempt_001' / (task['run_id'] + '.json'),
                   dict(status='FAIL', scientific_fit_started=False, actual_epochs=0, history=[]))
        progress = self.here / 'slurm_logs/pilot/attempt_001' / task['run_id'] / 'progress.json'
        for flag in (True, None):
            with self.subTest(flag=flag):
                self.write(progress, dict(status='SETUP', scientific_fit_started=flag, actual_epochs=0))
                with self.assertRaises(ValueError):
                    submit.retry_review()
        self.write(progress, dict(status='SETUP', scientific_fit_started=False, actual_epochs=0))
        self.assertEqual(submit.retry_review(), sha(path))

    def test_nonterminal_parent_rejected(self):
        path, review = self.prepare_retry()
        self.write(path, dict(review, scheduler=dict(JobIDRaw='12345', State='RUNNING')))
        with self.assertRaises(ValueError):
            submit.retry_review()


class ConfirmationTests(FileFixture):
    def prepare_confirmation(self, scores=None):
        self.stack.enter_context(patch.object(c, 'STAGE', 'confirmation'))
        source, execution = 'b' * 64, 'e' * 40
        self.stack.enter_context(patch.object(provenance, 'verify', return_value={'source_hash': source}))
        plan = self.write(self.here / 'study_plan.json', dict(confirmation='absolute >= both'))
        plan_hash = sha(plan)
        scores = scores or dict(baseline_dual=.0633, relative_phase=.0626, absolute_phase=.0633)
        results, hashes = {}, {}
        pairing = dict(initial_backbone_sha256='i' * 64, initial_common_calibrator_hashes={},
                       rng_components={'torch': 'j' * 64}, protocol={'dataset': 'fixed'},
                       manifest_sha256='m' * 64, train_time_stats_sha256='t' * 64,
                       verified_history_stats={}, precision={'amp': False},
                       optimizer_settings={'lr': .001}, first_train_batch_sha256='f' * 64)
        for mode in c.MODES:
            raw = dict(pairing, status='PASS', seed=2026, phase_mode=mode, mode='dual',
                       study_id=c.STUDY, study_phase='pilot', source_hash=source,
                       execution_commit=execution, plan_sha256=plan_hash,
                       scientific_fit_started=True, scientific_fit_completed=True,
                       actual_epochs=12, TEST='NOT_RUN', test_evaluation_count=0,
                       best_valid_metrics={'ndcg@10': scores[mode]})
            path = self.write(self.here / 'runs/pilot/attempt_001' / (mode + '.json'), raw)
            results[mode] = dict(path=str(path.relative_to(self.root)), sha256=sha(path))
            hashes[mode] = sha(path)
        audit = dict(status='PASS', study_id=c.STUDY, study_phase='pilot', source_hash=source,
                     execution_commit=execution, plan_sha256=plan_hash,
                     scientific_fits_started=3, scientific_fits_completed=3,
                     unknown_scientific_starts=0, TEST='NOT_RUN', test_evaluation_count=0,
                     pairing_verified=True, pilot_result_sha256=hashes)
        audit_path = self.write(self.here / 'evidence/pilot/independent_audit.json', audit)
        decision = dict(status='AUTHORIZED_BY_FROZEN_RULE', study_id=c.STUDY,
                        source_hash=source, execution_commit=execution, plan_sha256=plan_hash,
                        audit_path=str(audit_path.relative_to(self.root)), audit_sha256=sha(audit_path),
                        pilot_results=results, pilot_scores=scores, pairing_verified=True)
        path = self.write(self.here / 'runtime/confirmation_decision.json', decision)
        return path, decision, audit_path, audit

    def rewrite_audit(self, decision_path, decision, audit_path, audit):
        self.write(audit_path, audit)
        decision['audit_sha256'] = sha(audit_path)
        self.write(decision_path, decision)

    def test_equality_allowed_but_not_labelled_improvement(self):
        path, _, _, _ = self.prepare_confirmation()
        self.assertEqual(provenance.confirmation_authorization(), sha(path))

    def test_absolute_below_either_comparator_blocks(self):
        path, decision, audit_path, audit = self.prepare_confirmation()
        originals = {mode: json.loads((self.root / entry['path']).read_text())
                     for mode, entry in decision['pilot_results'].items()}
        for comparator in ('baseline_dual', 'relative_phase'):
            with self.subTest(comparator=comparator):
                changed = copy.deepcopy(decision)
                changed_audit = copy.deepcopy(audit)
                for mode, entry in changed['pilot_results'].items():
                    self.write(self.root / entry['path'], originals[mode])
                changed['pilot_scores'][comparator] = .0634
                entry = changed['pilot_results'][comparator]
                raw_path = self.root / entry['path']
                raw = json.loads(raw_path.read_text())
                raw['best_valid_metrics']['ndcg@10'] = .0634
                self.write(raw_path, raw)
                entry['sha256'] = sha(raw_path)
                changed_audit['pilot_result_sha256'][comparator] = entry['sha256']
                self.rewrite_audit(path, changed, audit_path, changed_audit)
                with self.assertRaisesRegex(ValueError, 'metric condition'):
                    provenance.confirmation_authorization()

    def test_modified_raw_bytes_without_rebound_hash_rejected(self):
        _, decision, _, _ = self.prepare_confirmation()
        raw_path = self.root / decision['pilot_results']['absolute_phase']['path']
        raw = json.loads(raw_path.read_text())
        self.write(raw_path, dict(raw, seed=2027))
        with self.assertRaises(ValueError):
            provenance.confirmation_authorization()

    def test_invalid_metric_cannot_authorize_even_with_rebound_hashes(self):
        path, decision, audit_path, audit = self.prepare_confirmation()
        entry = decision['pilot_results']['absolute_phase']
        raw_path = self.root / entry['path']
        original = json.loads(raw_path.read_text())
        for value in (True, -1.0, 1.1, 'infinite'):
            with self.subTest(value=value):
                raw = copy.deepcopy(original)
                raw['best_valid_metrics']['ndcg@10'] = 9.0 if value == 'infinite' else value
                self.write(raw_path, raw)
                if value == 'infinite':
                    # Legal JSON number, overflowing to inf when parsed as float.
                    raw_path.write_text(raw_path.read_text().replace('"ndcg@10": 9.0', '"ndcg@10": 1e999'))
                entry['sha256'] = sha(raw_path)
                changed_audit = copy.deepcopy(audit)
                changed_audit['pilot_result_sha256']['absolute_phase'] = entry['sha256']
                self.rewrite_audit(path, decision, audit_path, changed_audit)
                with self.assertRaisesRegex(ValueError, 'Invalid pilot metric'):
                    provenance.confirmation_authorization()

    def test_unrelated_pass_audit_does_not_authorize(self):
        path, decision, audit_path, audit = self.prepare_confirmation()
        for field, value in (('source_hash', 'c' * 64), ('execution_commit', 'd' * 40),
                             ('plan_sha256', 'a' * 64), ('pairing_verified', False),
                             ('scientific_fits_started', 2), ('scientific_fits_completed', 2),
                             ('unknown_scientific_starts', 1), ('test_evaluation_count', 1),
                             ('pilot_result_sha256', {})):
            with self.subTest(field=field):
                self.rewrite_audit(path, decision, audit_path, dict(audit, **{field: value}))
                with self.assertRaises(ValueError):
                    provenance.confirmation_authorization()

    def test_raw_pairing_or_source_drift_cannot_hide_in_decision_boolean(self):
        path, decision, audit_path, audit = self.prepare_confirmation()
        entry = decision['pilot_results']['absolute_phase']
        raw_path = self.root / entry['path']
        original = json.loads(raw_path.read_text())
        for field, value in (('source_hash', 'a' * 64), ('execution_commit', 'a' * 40),
                             ('scientific_fit_started', False), ('test_evaluation_count', 1),
                             ('first_train_batch_sha256', 'a' * 64)):
            with self.subTest(field=field):
                self.write(raw_path, dict(original, **{field: value}))
                entry['sha256'] = sha(raw_path)
                changed_audit = copy.deepcopy(audit)
                changed_audit['pilot_result_sha256']['absolute_phase'] = entry['sha256']
                self.rewrite_audit(path, decision, audit_path, changed_audit)
                with self.assertRaises(ValueError):
                    provenance.confirmation_authorization()


if __name__ == '__main__':
    unittest.main()
