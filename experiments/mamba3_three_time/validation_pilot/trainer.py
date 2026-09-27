"""Reuse safe trainer/fit semantics; add detached three-path VALID diagnostics."""
import math
import time
import torch
from experiments.mamba3_context_time.trainer import trainer_class as safe_trainer, deadline_check
from .diagnostics import Diagnostics
from .provenance import atomic_json, assert_upstream


def trainer_class(base, valid_loader, record, paths, expected_first_batch=None):
    parent = safe_trainer(base, valid_loader, record, paths, expected_first_batch)

    class PilotTrainer(parent):
        def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
            assert_upstream(self.model)
            torch.cuda.synchronize()
            start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            result = super()._train_epoch(train_data, epoch_idx, loss_func, show_progress)
            torch.cuda.synchronize()
            self.training_row.update(train_seconds=time.perf_counter()-start,
                                     train_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                     train_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            return result

        def _valid_epoch(self, valid_data, show_progress=False):
            if valid_data is not valid_loader:
                raise ValueError('Only the reserved VALID loader is permitted')
            assert_upstream(self.model)
            deadline_check()
            collector = Diagnostics(record['mode'])
            hook = self.model.times.register_forward_hook(collector.hook)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            try:
                # Keep upstream score/metric semantics, bypass the unrelated context-time collector.
                score, metrics = base._valid_epoch(self, valid_data, show_progress=show_progress)
            finally:
                hook.remove()
            torch.cuda.synchronize()
            if not math.isfinite(float(score)) or not all(math.isfinite(float(v)) for v in metrics.values()):
                raise ValueError('Nonfinite VALID result')
            record['history'].append(dict(epoch=self.current_epoch, valid_ndcg10=float(score),
                valid_metrics=dict(metrics), valid_seconds=time.perf_counter()-start,
                valid_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                valid_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                diagnostics=collector.result(), **self.training_row))
            record['actual_epochs'] = len(record['history'])
            atomic_json(paths['result'], record)
            return score, metrics

    return PilotTrainer
