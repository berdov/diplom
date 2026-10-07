"""Extra stdlib regressions for the reviewed transition; outside frozen 100 tests.

Only temporary synthetic manifests/evidence are written. No Git, SSH, scheduler,
Torch, model, or checkpoint loading. The expensive source-byte verifier is
injected; the actual authorization, manifest lineage, and JSON readers run.
"""
import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from experiments.mamba3_absolute_phase import config as c, provenance, report
from experiments.mamba3_mimo_time.records import digest, read, sha


class ConfirmationLineageTests(unittest.TestCase):
    def setUp(self):
        original = read(c.HERE/'source_manifest.json')
        plan = (c.HERE/'study_plan.json').read_bytes()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.here = self.root/'experiments/mamba3_absolute_phase'
        self.here.mkdir(parents=True)
        (self.here/'study_plan.json').write_bytes(plan)
        self.manifest_path = self.here/'source_manifest_confirmation.json'
        for name, value in dict(ROOT=self.root, HERE=self.here, MANIFEST=self.manifest_path,
                                STAGE='confirmation', EXECUTION_ATTEMPT='001').items():
            self.stack.enter_context(patch.object(c, name, value))
        self.old = copy.deepcopy(original)
        self.new = copy.deepcopy(original)
        for i, path in enumerate(provenance.CONFIRMATION_LINEAGE_PATHS):
            self.new['files'][path] = digest(dict(synthetic_changed_path=path, index=i))
        self.pilot_manifest_path = self.here/'source_manifest.json'
        self.write(self.pilot_manifest_path, self.old)
        self.failure_path = self.here/'evidence/confirmation_cpu_scope_failure/failed_cpu.json'
        self.failure = dict(status='FAIL', study_phase='confirmation', scientific_fits=0,
            execution_commit='1'*40, source_hash=self.old['source_hash'],
            TEST='NOT_RUN', test_evaluation_count=0, mimo_model_forward_calls=0,
            cpu_tests=dict(run=100, failures=29, errors=2, skipped=0))
        self.review_path = self.here/'runtime/confirmation_transition_review.json'
        self.review = dict(status='PASS', study_id=c.STUDY, pilot_execution_commit='1'*40,
            confirmation_execution_commit='2'*40, pilot_source_hash=self.old['source_hash'],
            plan_sha256=sha(self.here/'study_plan.json'),
            changed_paths=list(provenance.CONFIRMATION_LINEAGE_PATHS))
        self.lineage_path = self.here/'runtime/confirmation_source_lineage.json'
        self.lineage = dict(self.review, schema='absolute_phase_test_scope_lineage_v1',
            reason='confirmation_test_fixture_scope', pilot_manifest_path=self.rel(self.pilot_manifest_path),
            pilot_manifest_sha256=sha(self.pilot_manifest_path),
            confirmation_manifest_path=self.rel(self.manifest_path),
            failure_evidence_path=self.rel(self.failure_path), review_path=self.rel(self.review_path))
        self.lineage.pop('changed_paths')
        self.decision_path = self.here/'runtime/confirmation_decision.json'
        self.decision = dict(status='AUTHORIZED_BY_FROZEN_RULE', study_id=c.STUDY,
            execution_commit='1'*40, confirmation_execution_commit='2'*40,
            source_hash=self.old['source_hash'], plan_sha256=sha(self.here/'study_plan.json'),
            source_lineage_path=self.rel(self.lineage_path), pairing_verified=True)
        scores = dict(baseline_dual=.0633, relative_phase=.0616, absolute_phase=.0635)
        self.decision['pilot_scores'] = scores
        self.decision['pilot_results'] = {}
        self.raw = {}
        for mode in c.MODES:
            raw = dict.fromkeys(report.PAIRING, 'identical synthetic pairing metadata')
            raw.update(status='PASS', seed=2026, phase_mode=mode, scientific_fit_started=True,
                study_id=c.STUDY, study_phase='pilot', execution_commit='1'*40,
                source_hash=self.old['source_hash'], plan_sha256=self.decision['plan_sha256'],
                TEST='NOT_RUN', test_evaluation_count=0, best_valid_metrics={'ndcg@10':scores[mode]})
            path = self.here/f'runs/pilot/attempt_001/{mode}.json'
            self.write(path, raw)
            self.raw[mode] = path
            self.decision['pilot_results'][mode] = dict(path=self.rel(path), sha256=sha(path))
        audit_path = self.here/'evidence/pilot/independent_audit.json'
        audit = dict(status='PASS', study_id=c.STUDY, study_phase='pilot', execution_commit='1'*40,
            source_hash=self.old['source_hash'], plan_sha256=self.decision['plan_sha256'],
            scientific_fits_started=3, scientific_fits_completed=3, unknown_scientific_starts=0,
            TEST='NOT_RUN', test_evaluation_count=0, pairing_verified=True,
            pilot_result_sha256={mode:sha(path) for mode,path in self.raw.items()})
        self.write(audit_path, audit)
        self.decision.update(audit_path=self.rel(audit_path), audit_sha256=sha(audit_path))
        self.rebind()
        self.stack.enter_context(patch.object(provenance, 'verify', side_effect=lambda:self.new))

    def rel(self, path):
        return str(path.relative_to(self.root))

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

    def rebind(self):
        self.new['source_hash'] = digest(self.new['files'])
        for value in (self.lineage,self.review,self.decision):
            value['confirmation_source_hash'] = self.new['source_hash']
        self.write(self.manifest_path,self.new)
        self.lineage['confirmation_manifest_sha256'] = sha(self.manifest_path)
        changed = sorted(k for k in self.old['files'] if self.old['files'][k] != self.new['files'].get(k))
        self.lineage['changes'] = [dict(path=k,before_sha256=self.old['files'][k],after_sha256=self.new['files'].get(k)) for k in changed]
        unchanged = {k:self.old['files'][k] for k in sorted(self.old['files']) if k not in changed}
        self.lineage['unchanged_file_count'] = len(unchanged)
        self.lineage['unchanged_files_sha256'] = digest(unchanged)
        self.write(self.failure_path,self.failure)
        self.lineage['failure_evidence_sha256'] = sha(self.failure_path)
        self.write(self.review_path,self.review)
        self.lineage['review_sha256'] = sha(self.review_path)
        self.write(self.lineage_path,self.lineage)
        self.decision['source_lineage_sha256'] = sha(self.lineage_path)
        self.write(self.decision_path,self.decision)

    def authorize(self):
        return provenance.confirmation_authorization('2'*40)

    def test_exact_four_file_repair_preserves_original_pilot_authorization(self):
        self.assertEqual(self.authorize(),sha(self.decision_path))
        self.assertEqual(self.lineage['unchanged_file_count'],443)
        self.assertEqual(read(self.pilot_manifest_path),self.old)

    def test_current_execution_commit_must_match_review(self):
        with self.assertRaisesRegex(ValueError,'execution/source'):
            provenance.confirmation_authorization('3'*40)

    def test_scientific_file_change_rejected_even_with_rebound_metadata(self):
        path='experiments/mamba3_absolute_phase/phase.py'
        self.new['files'][path]='f'*64
        self.rebind()
        with self.assertRaisesRegex(ValueError,'exact four-file'):
            self.authorize()

    def test_unlisted_runtime_or_test_file_cannot_be_added(self):
        self.new['files']['experiments/mamba3_absolute_phase/runtime/hidden.py']='e'*64
        self.rebind()
        with self.assertRaisesRegex(ValueError,'path set'):
            self.authorize()

    def test_existing_manifest_file_cannot_be_removed(self):
        self.new['files'].pop('experiments/mamba3_absolute_phase/phase.py')
        self.rebind()
        with self.assertRaisesRegex(ValueError,'path set'):
            self.authorize()

    def test_all_four_changes_must_be_explicit(self):
        path=provenance.CONFIRMATION_LINEAGE_PATHS[0]
        self.new['files'][path]=self.old['files'][path]
        self.rebind()
        with self.assertRaisesRegex(ValueError,'exact four-file'):
            self.authorize()

    def test_review_bytes_cannot_change_after_binding(self):
        self.write(self.review_path,dict(self.review,comment='changed'))
        with self.assertRaisesRegex(ValueError,'review changed'):
            self.authorize()

    def test_review_scope_or_identity_cannot_hide_behind_new_sha(self):
        original=copy.deepcopy(self.review)
        for key,value in [('status','FAIL'),('changed_paths',[]),('pilot_source_hash','a'*64),
                          ('pilot_execution_commit','a'*40),('confirmation_execution_commit','a'*40),
                          ('plan_sha256','a'*64)]:
            with self.subTest(field=key):
                self.review=dict(original,**{key:value})
                self.rebind()
                with self.assertRaisesRegex(ValueError,'review scope/identity'):
                    self.authorize()

    def test_preserved_failure_identity_counts_and_no_fit_must_match(self):
        original=copy.deepcopy(self.failure)
        for key,value in [('status','PASS'),('study_phase','pilot'),('scientific_fits',1),
                          ('execution_commit','a'*40),('source_hash','a'*64),('test_evaluation_count',1),
                          ('mimo_model_forward_calls',1),('scientific_fits',False),
                          ('cpu_tests',dict(run=100,failures=0,errors=0,skipped=0))]:
            with self.subTest(field=key,value=value):
                self.failure=dict(original,**{key:value})
                self.rebind()
                with self.assertRaisesRegex(ValueError,'Unrelated CPU failure'):
                    self.authorize()

    def test_original_manifest_cannot_be_rewritten(self):
        changed=copy.deepcopy(self.old)
        changed['files']['experiments/mamba3_absolute_phase/model.py']='a'*64
        changed['source_hash']=digest(changed['files'])
        self.write(self.pilot_manifest_path,changed)
        self.lineage['pilot_manifest_sha256']=sha(self.pilot_manifest_path)
        self.rebind()
        with self.assertRaisesRegex(ValueError,'content/digest'):
            self.authorize()

    def test_plan_bytes_remain_frozen(self):
        (self.here/'study_plan.json').write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'source/plan changed'):
            self.authorize()

    def test_unchanged_table_digest_is_verified(self):
        self.lineage['unchanged_files_sha256']='a'*64
        self.write(self.lineage_path,self.lineage)
        self.decision['source_lineage_sha256']=sha(self.lineage_path)
        self.write(self.decision_path,self.decision)
        with self.assertRaisesRegex(ValueError,'changed/unchanged SHA'):
            self.authorize()

    def test_pilot_raw_evidence_cannot_be_relabelled_new_source(self):
        path=self.raw['absolute_phase']
        raw=read(path)
        raw['source_hash']=self.new['source_hash']
        self.write(path,raw)
        with self.assertRaisesRegex(ValueError,'Pilot result identity'):
            self.authorize()

    def test_lineage_symlink_is_rejected(self):
        alternate=self.lineage_path.with_name('alternate.json')
        alternate.write_bytes(self.lineage_path.read_bytes())
        self.lineage_path.unlink()  # Temporary synthetic fixture only.
        self.lineage_path.symlink_to(alternate)
        with self.assertRaisesRegex(ValueError,'Invalid lineage path'):
            self.authorize()

    def test_lineage_manifest_must_equal_the_verified_manifest(self):
        changed=dict(self.new,unexpected_metadata='not verified')
        self.write(self.manifest_path,changed)
        self.lineage['confirmation_manifest_sha256']=sha(self.manifest_path)
        self.write(self.lineage_path,self.lineage)
        self.decision['source_lineage_sha256']=sha(self.lineage_path)
        self.write(self.decision_path,self.decision)
        with self.assertRaisesRegex(ValueError,'content/digest'):
            self.authorize()


if __name__ == '__main__':
    unittest.main(verbosity=2)
