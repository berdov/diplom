"""Streaming TRAIN-history hidden states into deterministic MiniBatchKMeans."""

import numpy as np


def fit_stream(batches):
    from sklearn.cluster import MiniBatchKMeans

    estimator = MiniBatchKMeans(n_clusters=8, random_state=2026, batch_size=2048,
                               n_init=3, reassignment_ratio=0.01)
    count = 0
    batch_count = 0
    for values in batches:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 64 or not np.isfinite(values).all():
            raise ValueError('Expected finite hidden states [B,64]')
        if not count and len(values) < 8:
            raise ValueError('First KMeans batch must contain at least 8 histories')
        estimator.partial_fit(values)
        count += len(values)
        batch_count += 1
    if not count:
        raise ValueError('No TRAIN histories')
    return estimator.cluster_centers_, dict(train_histories=count, batches=batch_count,
                                            inertia_last_batch=float(estimator.inertia_))


def initialize_from_train(vanilla, train_dataset, config):
    import torch
    from torch.nn import functional as F

    vanilla.eval()
    vanilla.requires_grad_(False)
    def batches():
        with torch.no_grad():
            for start in range(0, len(train_dataset), 2048):
                batch = train_dataset.inter_feat[start:start + 2048].to(config['device'])
                # No target item or timestamp is passed to the frozen encoder.
                h = vanilla(batch[vanilla.ITEM_SEQ], batch[vanilla.ITEM_SEQ_LEN])
                yield F.normalize(h, dim=-1).float().cpu().numpy()
    return fit_stream(batches())
