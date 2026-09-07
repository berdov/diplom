"""Read existing VALID bytes; verify the original sidecar checksum in memory.

Lustre OST 0 is unavailable (2026-09-07), including the redundant ID sidecar.
No dataset/sidecar is created, copied, rewritten, or reprocessed by this helper.
"""
import csv
import hashlib
import io
from pathlib import Path


def validation_source_ids(summary):
    raw=Path(summary['valid_inter_path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=summary['files']['valid_inter_sha256']:
        raise RuntimeError('Existing VALID file SHA256 differs from frozen preparation manifest')
    ids=sorted(int(row['source_row_id:float']) for row in csv.DictReader(io.StringIO(raw.decode()),delimiter='\t'))
    serialized=('\n'.join(map(str,ids))+'\n').encode()
    expected=summary['files']['validation_source_row_ids_sha256']
    if hashlib.sha256(serialized).hexdigest()!=expected:
        raise RuntimeError('In-memory validation IDs differ from original frozen sidecar SHA256')
    if len(ids)!=summary['protocol_fingerprint']['validation'] or len(set(ids))!=len(ids):
        raise RuntimeError('Unexpected validation source ID cardinality')
    return set(ids)
