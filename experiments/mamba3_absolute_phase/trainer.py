"""Unchanged TRAIN/VALID fit; collect phase diagnostics in existing VALID forwards."""
import time
import torch
from . import config as c
from .diagnostics import PhaseDiagnostics, parameter_snapshot
from .progress import write as write_progress
from experiments.mamba3_mimo_time.trainer import trainer_class as mimo_trainer
from experiments.mamba3_mimo_time.records import read, update


def trainer_class(base, valid_loader, record, paths, expected_first_batch=None):
    parent = mimo_trainer(base, valid_loader, record, paths, expected_first_batch)
    coverage_summary = read(c.COVERAGE)['summary']

    class PhaseTrainer(parent):
        def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
            if record['phase_mode'] == 'baseline_dual':
                return super()._train_epoch(train_data, epoch_idx, loss_func, show_progress)
            counts = dict(backward_calls=0, finite=True, nonzero_calls=0)
            def observe_gradient(gradient):
                counts['backward_calls'] += 1
                if not bool(torch.isfinite(gradient).all()):
                    counts['finite'] = False
                    raise ValueError('Nonfinite phase W gradient')
                counts['nonzero_calls'] += int(bool(torch.count_nonzero(gradient)))
                return gradient
            hook = self.model.phase_adapter.W.register_hook(observe_gradient)
            try:
                result = super()._train_epoch(train_data, epoch_idx, loss_func, show_progress)
            finally:
                hook.remove()
            if counts['backward_calls'] == 0:
                raise ValueError('Phase W gradient hook was never called')
            self.training_row['phase_gradient_checks'] = dict(counts, hook_removed=True)
            return result

        def _valid_epoch(self, valid_data, show_progress=False):
            start = time.perf_counter()
            collector = PhaseDiagnostics(record['phase_mode']) if record['phase_mode'] != 'baseline_dual' else None
            if getattr(self.model, 'phase_observer', None) is not None:
                raise ValueError('Observer lifecycle collision')
            if collector is not None:
                self.model.phase_observer = collector.observe
            try:
                score, metrics = super()._valid_epoch(valid_data, show_progress)
            finally:
                self.model.phase_observer = None
            if collector is not None:
                observed = collector.result()
                if (observed.get('status') != 'MEASURED' or set(observed.get('layers', {})) != {'0', '1'}
                    or any(x['forward_calls'] <= 0 or x['histories'] != 23951 for x in observed['layers'].values())
                    or observed['layers']['0']['forward_calls'] != observed['layers']['1']['forward_calls']):
                    raise ValueError('Incomplete existing-VALID phase diagnostics')
                record['history'][-1]['diagnostics']['phase'] = dict(
                    observed=observed, parameters=parameter_snapshot(
                        self.model.phase_adapter, record['phase_mode'],
                        train_coverage_summary=coverage_summary))
                record['history'][-1]['valid_seconds'] = time.perf_counter() - start
            update(paths['result'], record)
            write_progress(record, paths)
            return score, metrics

        def _save_checkpoint(self, epoch, verbose=True, **kwargs):
            super()._save_checkpoint(epoch, verbose, **kwargs)
            meta = read(paths['metadata'])
            meta.update(phase_mode=record['phase_mode'],
                        phase=record['best_diagnostics'].get('phase'))
            update(paths['metadata'], meta)

    return PhaseTrainer
