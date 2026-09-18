"""Narrow RecBole adapter: keep fit/tie logic, replace checkpoint serialization."""
import math
import os
import time
from .diagnostics import ContextDiagnostics
from .provenance import atomic_json, save_state_dict, tensor_hash


def deadline_check():
    if time.time() >= float(os.environ.get('PIPELINE_DEADLINE', 'inf')):
        raise TimeoutError('Internal allocation deadline exceeded')


def trainer_class(base, valid_loader, record, paths, expected_first_batch=None):
    class ValidationTrainer(base):
        def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
            self.current_epoch = epoch_idx
            start = time.perf_counter()
            def observed_loss(interaction):
                deadline_check()
                if 'first_train_batch_sha256' not in record:
                    digest = tensor_hash(interaction.interaction)
                    record['first_train_batch_sha256'] = digest
                    atomic_json(paths['result'], record)
                    if expected_first_batch is not None and digest != expected_first_batch:
                        raise ValueError('Consumed first batch differs from replay')
                return self.model.calculate_loss(interaction)
            loss = super()._train_epoch(train_data, epoch_idx, observed_loss, show_progress)
            if not math.isfinite(float(loss)):
                raise ValueError('Nonfinite training loss')
            self.training_row = dict(train_loss=float(loss), train_seconds=time.perf_counter()-start)
            return loss

        def evaluate(self, eval_data, load_best_model=False, model_file=None, show_progress=False):
            if eval_data is not valid_loader or load_best_model or model_file is not None:
                raise ValueError('Only in-memory current-model VALID evaluation permitted')
            deadline_check()
            return super().evaluate(eval_data, load_best_model=False, show_progress=show_progress)

        def _full_sort_batch_eval(self, batched_data):
            deadline_check()
            return super()._full_sort_batch_eval(batched_data)

        def _valid_epoch(self, valid_data, show_progress=False):
            collector = ContextDiagnostics(record['mode'])
            self.model.diagnostic_collector = collector
            start = time.perf_counter()
            try:
                score, metrics = super()._valid_epoch(valid_data, show_progress=show_progress)
            finally:
                self.model.diagnostic_collector = None
            if not math.isfinite(float(score)) or not all(math.isfinite(float(v)) for v in metrics.values()):
                raise ValueError('Nonfinite VALID result')
            record['history'].append(dict(epoch=self.current_epoch, valid_ndcg10=float(score),
                valid_metrics=dict(metrics), valid_seconds=time.perf_counter()-start,
                diagnostics=collector.result(), **self.training_row))
            record['actual_epochs'] = len(record['history'])
            atomic_json(paths['result'], record)
            return score, metrics

        def _save_checkpoint(self, epoch, verbose=True, **kwargs):
            # Called by unchanged RecBole fit, including its last-equal-maximum semantics.
            row = record['history'][-1]
            if row['epoch'] != epoch or row['valid_ndcg10'] != float(self.best_valid_score):
                raise ValueError('Checkpoint callback/history mismatch')
            digest = save_state_dict(self.model, paths['checkpoint'])
            metadata = dict(epoch=epoch, epoch_indexing='zero-based', mode=record['mode'],
                seed=record['seed'], run_id=record['run_id'], execution_commit=record['execution_commit'],
                config_sha256=record['config_sha256'], source_hash=record['source_hash'],
                core_hash=record['core_hash'], metrics=row['valid_metrics'], checkpoint_sha256=digest)
            atomic_json(paths['metadata'], metadata)
            record.update(best_epoch=epoch, best_valid_score=row['valid_ndcg10'],
                best_valid_metrics=row['valid_metrics'], best_diagnostics=row['diagnostics'], checkpoint_sha256=digest)
            atomic_json(paths['result'], record)

    return ValidationTrainer
