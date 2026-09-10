"""Validation-only runtime guards, installed before importing dataset helpers."""
import os
from pathlib import Path
import re
import sys

TASK_ORDER = ('rank','is_click','long_view','is_like','is_profile_enter')
TEST_MARKER = re.compile(r'(^|[./_\-])test([./_\-]|$)', re.IGNORECASE)
CODE_SUFFIXES = {'.py', '.pyc', '.so', '.pyd', '.h', '.hpp', '.c', '.cpp'}


class DataAccessGuard:
    def __init__(self, immutable_roots):
        self.roots = tuple(Path(p).resolve() for p in immutable_roots)
        self.opened_data_paths = set()
        self.blocked_test_accesses = 0
        self.loader_requests = []

    def check_path(self, value, write=False):
        if isinstance(value, int) or value is None:
            return
        p = Path(os.fsdecode(value)).absolute()
        resolved = p.resolve()
        if p.suffix not in CODE_SUFFIXES and (TEST_MARKER.search(str(p)) or TEST_MARKER.search(str(resolved))):
            self.blocked_test_accesses += 1
            raise RuntimeError(f'TEST access forbidden: {p}')
        protected = any(resolved.is_relative_to(root) for root in self.roots)
        if protected and write:
            raise RuntimeError(f'Immutable shared data write forbidden: {resolved}')
        if protected:
            self.opened_data_paths.add(str(resolved))

    def audit(self, event, args):
        if event == 'open':
            mode, flags = args[1], args[2]
            write = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            self.check_path(args[0], write)
        elif event in {'os.remove','os.rmdir','os.mkdir','os.truncate','os.chmod'}:
            self.check_path(args[0], True)
        elif event in {'os.rename','os.link','os.symlink'}:
            self.check_path(args[0], True)
            self.check_path(args[1], True)

    def install(self):
        sys.addaudithook(self.audit)

    def loader_factory(self, original):
        def guarded(config, phase):
            if phase not in ('train','valid'):
                self.blocked_test_accesses += 1
                raise RuntimeError(f'Only train/valid dataloaders are allowed, got {phase}')
            self.loader_requests.append(phase)
            return original(config, phase)
        return guarded

    def record(self):
        return {'test_evaluation_count':0, 'blocked_test_accesses':self.blocked_test_accesses,
                'loader_requests':self.loader_requests, 'opened_shared_data_paths':sorted(self.opened_data_paths),
                'mechanism':'Python audit hook plus guarded RecBole loader factory; installed before dataset construction'}

