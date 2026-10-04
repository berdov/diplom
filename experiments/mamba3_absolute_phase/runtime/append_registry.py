"""Append audited individual phase runs; default is a read-only preview."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PREFIX_BYTES = 58287
PREFIX_SHA = '30ef3c9c58c90662367fe21de5fcc8d71315d6d475564ec0c6d9b1adb8053c63'
MODES = ('baseline_dual', 'relative_phase', 'absolute_phase')
STUDY = 'mamba3_absolute_phase_001'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    require(path.is_file() and not path.is_symlink(), 'Missing or symlinked evidence: ' + str(path))
    return json.loads(path.read_bytes())


def prepare(folder, registry, root=ROOT):
    folder, registry, root = Path(folder).resolve(), Path(registry).resolve(), Path(root).resolve()
    require(folder.is_relative_to(root) and registry.is_relative_to(root), 'Publication paths must stay inside repository')
    audit = read(folder / 'independent_audit.json')
    manifest_path = folder / 'preservation_manifest.json'
    saved = read(manifest_path)
    phase, attempt = saved['study_phase'], saved['execution_attempt']
    require(phase in ('pilot', 'confirmation') and attempt in ('001', '002'), 'Unexpected phase/attempt')
    seeds = (2026,) if phase == 'pilot' else (2027, 2028, 2029, 2030)
    expected = dict(status='PASS', study_id=STUDY, study_phase=phase, execution_attempt=attempt,
                    scientific_fits_started=3 * len(seeds), scientific_fits_completed=3 * len(seeds),
                    unknown_scientific_starts=0, pairing_verified=True, TEST='NOT_RUN', test_evaluation_count=0)
    require(all(audit.get(k) == v for k, v in expected.items()), 'Complete independent audit required')
    require(audit['preservation_manifest_sha256'] == sha(manifest_path.read_bytes()), 'Preservation manifest changed after audit')
    require(all(audit[k] == saved[k] for k in ('study_phase', 'execution_attempt', 'execution_commit', 'source_hash', 'job_id')), 'Audit/preservation identity mismatch')
    inventory = {row['path']: row for row in saved['files']}
    require(len(inventory) == len(saved['files']), 'Duplicate preserved paths')

    def preserved(relative):
        rel = Path(relative)
        path = folder / 'files' / rel
        require(not rel.is_absolute() and '..' not in rel.parts and path.resolve().is_relative_to(folder / 'files'), 'Unsafe preserved path')
        value = read(path)
        entry = inventory[relative]
        require(path.stat().st_size == entry['bytes'] and sha(path.read_bytes()) == entry['sha256'], 'Preserved bytes changed: ' + relative)
        return path, value

    source_name = 'source_manifest.json' if attempt == '001' else 'source_manifest_002.json'
    source_path, source = preserved(source_name)
    digest = sha(json.dumps(source['files'], sort_keys=True, separators=(',', ':')).encode())
    current_source = root / 'experiments/mamba3_absolute_phase' / source_name
    read(current_source)
    require(source['source_hash'] == audit['source_hash'] == digest and
            len(source['files']) == audit['source_blobs'] and sha(source_path.read_bytes()) == sha(current_source.read_bytes()), 'Frozen source identity mismatch')
    plan_path, _ = preserved('study_plan.json')
    require(sha(plan_path.read_bytes()) == audit['plan_sha256'], 'Frozen plan identity mismatch')
    original = registry.read_bytes()
    require(sha(original[:PREFIX_BYTES]) == PREFIX_SHA and original.endswith(b'\n'), 'Original registry prefix or newline changed')
    reader = csv.DictReader(io.StringIO(original.decode('utf-8')))
    existing, header = list(reader), reader.fieldnames
    require(len(header) == 41 and len(list(csv.DictReader(io.StringIO(original[:PREFIX_BYTES].decode())))) == 125, 'Original registry structure changed')
    require(all(None not in row and all(value is not None for value in row.values()) for row in existing), 'Malformed existing CSV row')
    tasks = [(mode, seed, f'mamba3_absolute_phase_{mode}_seed{seed}_001') for seed in seeds for mode in MODES]
    require(set(audit['result_sha256']) == {run for _, _, run in tasks}, 'Audited individual run set differs')
    rows, missing = [], []
    for mode, seed, run_id in tasks:
        path, raw = preserved(f'runs/{phase}/attempt_{attempt}/{run_id}.json')
        require(sha(path.read_bytes()) == audit['result_sha256'][run_id], 'Raw run changed after audit')
        identity = dict(status='PASS', study_id=STUDY, study_phase=phase, execution_attempt=attempt,
                        phase_mode=mode, mode='dual', seed=seed, run_id=run_id,
                        scientific_fit_started=True, TEST='NOT_RUN', test_evaluation_count=0)
        require(all(raw.get(k) == v for k, v in identity.items()) and all(raw[k] == audit[k] for k in
                ('execution_commit', 'source_hash', 'plan_sha256', 'job_id')), 'Raw scientific identity mismatch')
        require(raw['source_manifest_sha256'] == sha(source_path.read_bytes()), 'Raw/source manifest binding mismatch')
        metrics = raw['best_valid_metrics']
        require(set(metrics) == {f'{kind}@{k}' for kind in ('hit', 'recall', 'ndcg') for k in (5, 10, 20, 50)}, 'Expected exactly twelve metrics')
        require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 and round(v, 4) == v for v in metrics.values()), 'Invalid metric precision/range')
        require(type(raw['best_epoch']) is int and type(raw['actual_epochs']) is int and 0 <= raw['best_epoch'] < raw['actual_epochs'] <= 300, 'Invalid epoch counts')
        row = dict.fromkeys(header, '')
        row.update(record_type='experiment', source='ours', run_id=run_id, model='AbsolutePhaseMamba3Rec',
                   model_variant='MIMO_dual_' + mode, dataset='KuaiRand', protocol='B', split='validation',
                   evaluation='full_7111_items', status='completed', seed=str(seed), train_candidates='full_softmax',
                   item_universe='7111', best_epoch=str(raw['best_epoch']), actual_epochs=str(raw['actual_epochs']),
                   validation_ndcg10=format(metrics['ndcg@10'], '.4f'), test_evaluation_count='0',
                   git_commit=raw['execution_commit'], notes_path='reports/MAMBA3_TIME_MECHANISMS_RESULTS.md#absolute-phase-' + phase,
                   test_used='no', source_json=path.relative_to(root).as_posix())
        row.update({f'{label}@{k}': format(metrics[f'{kind}@{k}'], '.4f') for label, kind in
                    (('HR', 'hit'), ('Recall', 'recall'), ('NDCG', 'ndcg')) for k in (5, 10, 20, 50)})
        found = [old for old in existing if old['run_id'] == run_id]
        require(not found or found == [row], 'Existing scientific row conflicts: ' + run_id)
        rows.append(row)
        if not found:
            missing.append(row)
    buffer = io.StringIO(newline='')
    csv.DictWriter(buffer, fieldnames=header, lineterminator='\n').writerows(missing)
    payload = buffer.getvalue().encode('utf-8')
    return original, payload, dict(status='DRY_RUN', study_phase=phase, audited_rows=len(rows),
        rows_before=len(existing), rows_to_append=len(missing), rows_after=len(existing) + len(missing),
        original_sha256=sha(original), resulting_sha256=sha(original + payload),
        audit_sha256=sha((folder / 'independent_audit.json').read_bytes()), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('--write', action='store_true', help='Append only after the complete read-only validation')
    args = parser.parse_args()
    registry = ROOT / 'experiments/results.csv'
    original, payload, preview = prepare(args.evidence, registry)
    if args.write and payload:
        import fcntl
        import os
        with registry.open('r+b') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            require(handle.read() == original, 'Registry changed since validation; refusing append')
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        require(registry.read_bytes() == original + payload, 'Registry append verification failed')
    if args.write:
        preview['status'] = 'APPENDED' if payload else 'ALREADY_PRESENT'
    print(json.dumps(preview, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
