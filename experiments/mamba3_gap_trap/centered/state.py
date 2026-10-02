"""Common state mapping; unchanged parameter/RNG hashes and pair contract."""
from types import SimpleNamespace
from .. import state as parent
from . import config as c
from .reuse import bind

paired = parent.paired
effective_check = bind(parent, {'c': c})['effective_check']


def view(model):
    return SimpleNamespace(state_dict=model.state_dict, load_state_dict=model.load_state_dict,
                           times=model.times, gap_trap_mode='gap_trap' if model.gap_trap_mode == 'centered_gap_trap' else model.gap_trap_mode)


def initial(model, loader=None):
    return parent.initial(view(model), loader)


def transfer_common(source, target):
    return parent.transfer_common(view(source), view(target))
