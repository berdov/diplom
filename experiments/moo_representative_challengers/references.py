"""Fetch pinned reference files outside the repository; never vendor source."""
import argparse
import hashlib
import os
from pathlib import Path
import urllib.request
import yaml

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get('MOO_REFERENCE_CACHE', '~/.cache/diplom-moo-references')).expanduser()


def reference_path(method, relative):
    upstream = yaml.safe_load((HERE / 'provenance.yaml').read_text())['methods'][method]['upstream']
    path = CACHE / upstream['repository'].split('/')[-1] / relative
    expected = upstream['files_sha256'][relative]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Missing or changed pinned reference: {path}; run references --fetch')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true', required=True)
    parser.parse_args()
    if CACHE.resolve().is_relative_to(HERE.parents[1]):
        raise RuntimeError('Reference cache must be outside repository')
    for spec in yaml.safe_load((HERE / 'provenance.yaml').read_text())['methods'].values():
        up = spec['upstream']
        for relative, expected in up['files_sha256'].items():
            path = CACHE / up['repository'].split('/')[-1] / relative
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
                continue
            url = f"https://raw.githubusercontent.com/{up['repository']}/{up['commit']}/{relative}"
            content = urllib.request.urlopen(url, timeout=30).read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise RuntimeError(f'Upstream SHA256 mismatch: {url}')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    print('Pinned reference files verified')


if __name__ == '__main__':
    main()
