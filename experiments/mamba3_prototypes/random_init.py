"""Data-independent initialization; never consumes the training RNG stream."""

import torch


def random_prototypes(seed=2026, std=.02):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return torch.empty(8, 64).normal_(mean=0, std=std, generator=generator)


def metadata():
    return dict(prototype_initialization='random_normal', prototype_random_init_std=.02,
                prototype_random_seed=2026, K=8, temperature=1.0, test_evaluation_count=0,
                initialization='data-independent Normal(0, .02)',
                reference_commit='bbb9cd954e929df0e4ebc737be41b8440ecdb0bc',
                intended_difference='initial P only', train_statistics_used=False,
                frozen_encoder_used=False)
