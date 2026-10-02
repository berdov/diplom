"""Same inherited environment policy, with explicit new stage validation."""
from ..process_env import child_environment as parent_environment


def child_environment(stage, parent):
    if stage not in ('preflight','gate','smoke','fixed_replay','centered_gap_trap'):
        raise ValueError('Unplanned centered stage')
    return parent_environment('gap_trap' if stage == 'centered_gap_trap' else stage, parent)
