import builtins
import copy
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.mamba3_context_time import report
from experiments.mamba3_context_time.config import plan
from experiments.mamba3_context_time.provenance import atomic_json, sha
from experiments.mamba3_context_time.tests.test_guards import records

SVG = {'s': 'http://www.w3.org/2000/svg'}


@pytest.fixture
def reporting(tmp_path, monkeypatch):
    (tmp_path / 'runs').mkdir()
    rows = records()
    for row in rows.values():
        row.update(execution_commit='a' * 40, job_id='report-fixture',
                   initial_backbone_sha256='same-backbone', rng_before_fit_sha256='same-rng',
                   first_train_batch_sha256='same-batch')

    def local_paths(task):
        runtime = tmp_path / 'runtime' / task['mode']
        return dict(runtime=runtime, result=tmp_path / 'runs' / (task['run_id'] + '.json'),
                    checkpoint=runtime / 'checkpoint.pth', metadata=runtime / 'metadata.json',
                    lock=runtime / 'run.lock')

    def publish(mode=None):
        for task in plan()['tasks']:
            if mode is not None and task['mode'] != mode:
                continue
            row = rows[task['mode']]
            p = local_paths(task)
            p['runtime'].mkdir(parents=True, exist_ok=True)
            # Opaque hash fixture, not a loadable model or a scientific result.
            p['checkpoint'].write_bytes(b'reporting test fixture only')
            row['checkpoint_sha256'] = sha(p['checkpoint'])
            atomic_json(p['result'], row)

    monkeypatch.setattr(report, 'HERE', tmp_path)
    monkeypatch.setattr(report, 'SUMMARY', tmp_path / 'runs/pilot_summary.json')
    monkeypatch.setattr(report, 'paths', local_paths)
    monkeypatch.setattr(report, 'verify', lambda: dict(source_hash='fixture', files={}))
    return SimpleNamespace(root=tmp_path, rows=rows, paths=local_paths, publish=publish)


def test_report_without_matplotlib(reporting, monkeypatch):
    reporting.publish()
    original = builtins.__import__

    def no_plot_packages(name, *args, **kwargs):
        if name.split('.')[0] in ('matplotlib', 'svgwrite', 'PIL', 'cairo', 'plotly', 'seaborn'):
            raise ImportError('Plot packages unavailable in this fixture')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_plot_packages)
    report.main()
    result = json.loads(report.SUMMARY.read_text())
    assert result['status'] == 'COMPLETE' and result['plot_status'] == 'PASS'
    svg = ET.parse(reporting.root / 'runs/pilot_summary.svg').getroot()
    assert svg.tag == '{http://www.w3.org/2000/svg}svg'
    text = ' '.join(svg.itertext())
    for label in ('VALID NDCG@10', 'seed2026', 'Полный запуск', 'Первые 27 эпох', 'Historical separate'):
        assert label in text
    assert all(t['mode'] in text for t in plan()['tasks'])
    assert not svg.findall('.//s:script', SVG) and not svg.findall('.//s:image', SVG)
    assert '![VALID NDCG@10](pilot_summary.svg)' in (reporting.root / 'runs/pilot_summary.md').read_text()


def test_svg_coordinates_match_summary(tmp_path):
    raw = records()
    for row in raw.values():
        history = []
        for epoch in range(30):
            h = copy.deepcopy(row['history'][0])
            h['epoch'] = epoch
            value = row['best_valid_score'] - (.001 if epoch < 27 else 0)
            h['valid_metrics']['ndcg@10'] = h['valid_ndcg10'] = value
            history.append(h)
        row.update(history=history, actual_epochs=30, best_epoch=29)
    summary = report.summarize(raw, 'fixture')
    summary['historical'] = dict(ndcg10=.0633)
    report.render(summary, tmp_path / 'plot.svg')
    svg = ET.parse(tmp_path / 'plot.svg').getroot()
    points = svg.findall('.//*[@data-mode][@data-series]')
    assert len(points) == 10
    values = [.0633] + [r[k] for r in summary['rows'] for k in ('ndcg10', 'first27')]
    pad = max((max(values) - min(values)) * .15, .001)
    low, high = min(values) - pad, max(values) + pad
    for i, row in enumerate(summary['rows']):
        for series, key in (('full', 'ndcg10'), ('first27', 'first27')):
            node = next(n for n in points if n.get('data-mode') == row['mode'] and n.get('data-series') == series)
            assert float(node.get('data-value')) == row[key]
            x = float(node.get('cx') if series == 'full' else node.get('data-cx'))
            y = float(node.get('cy') if series == 'full' else node.get('data-cy'))
            assert x == pytest.approx(100 + (i + .5) * 940 / 5)
            assert y == pytest.approx(360 - (row[key] - low) / (high - low) * 280)
            if series == 'first27':
                coords = [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', node.get('d'))]
                assert coords[:2] == pytest.approx([x - 5, y - 5])
    historical = svg.find('.//*[@data-series="historical"]')
    assert float(historical.get('data-value')) == .0633 and historical.get('stroke-dasharray')


@pytest.mark.parametrize('kind', ('partial', 'failed', 'empty'))
def test_missing_runs_have_status_not_zero_points(reporting, kind):
    if kind != 'empty':
        for i, row in enumerate(reporting.rows.values()):
            row['status'] = 'PASS' if kind == 'partial' and i == 0 else ('FAIL' if i % 2 else 'NOT_RUN')
            row['error'] = '<fixture & failure>'
        reporting.publish()
    report.main()
    result = json.loads(report.SUMMARY.read_text())
    assert result['status'] == 'INCOMPLETE' and result['plot_status'] == 'PASS'
    svg = ET.parse(reporting.root / 'runs/pilot_summary.svg').getroot()
    points = svg.findall('.//*[@data-mode][@data-series]')
    assert len(points) == (2 if kind == 'partial' else 0)
    statuses = svg.findall('.//s:text[@data-mode]', SVG)
    assert len(statuses) == (4 if kind == 'partial' else 5)
    assert all(n.text in ('FAIL', 'NOT_RUN') for n in statuses)
    assert all(float(n.get('data-value')) > 0 for n in points)
    for node in svg.iter():
        for key in ('x', 'y', 'x1', 'x2', 'y1', 'y2', 'cx', 'cy', 'data-cx', 'data-cy'):
            if key in node.attrib:
                assert math.isfinite(float(node.get(key)))


def test_svg_escapes_status_text(tmp_path):
    summary = dict(rows=[dict(mode='routed', status='FAIL <fixture & detail>')], historical=dict(ndcg10=.0633))
    report.render(summary, tmp_path / 'plot.svg')
    svg = ET.parse(tmp_path / 'plot.svg').getroot()
    assert svg.find('.//s:text[@data-mode="routed"]', SVG).text == 'FAIL <fixture & detail>'


def test_renderer_failure_preserves_numeric_reports(reporting, monkeypatch, capsys):
    reporting.publish()
    before = {p: p.read_bytes() for p in (reporting.root / 'runs').glob('*.json')}

    def fail(result, output):
        assert json.loads(report.SUMMARY.read_text())['status'] == 'COMPLETE'
        assert '| dense11 | 2026 |' in (reporting.root / 'runs/pilot_summary.md').read_text()
        raise OSError('injected SVG failure')

    monkeypatch.setattr(report, 'render', fail)
    report.main()
    result = json.loads(report.SUMMARY.read_text())
    md = (reporting.root / 'runs/pilot_summary.md').read_text()
    assert result['status'] == 'COMPLETE' and result['plot_status'] == 'FAILED'
    assert result['plot_error'] == "OSError('injected SVG failure')"
    assert 'Изображение не создано' in md and 'pilot_summary.svg' not in md
    assert '| dense11 | 2026 | 611214 | 0.0610 |' in md
    assert result['rows'][1]['ndcg10'] == .061
    assert all(p.read_bytes() == value for p, value in before.items())
    assert 'Предупреждение' in capsys.readouterr().err


@pytest.mark.parametrize('broken', ('source', 'config', 'checkpoint', 'history', 'best', 'rng', 'nan', 'test', 'read'))
def test_invalid_scientific_data_is_not_plot_warning(reporting, monkeypatch, broken):
    row = reporting.rows['dense11']
    if broken == 'source': row['source_hash'] = 'wrong'
    if broken == 'config': row['config']['learning_rate'] = .5
    if broken == 'history': row['actual_epochs'] += 1
    if broken == 'best': row['best_epoch'] = 0
    if broken == 'rng': row['rng_before_fit_sha256'] = 'wrong'
    if broken == 'test': row['test_evaluation_count'] = 1
    reporting.publish()
    path = reporting.paths(next(t for t in plan()['tasks'] if t['mode'] == 'dense11'))
    if broken == 'checkpoint': path['checkpoint'].write_bytes(b'changed')
    if broken == 'read': path['result'].write_text('{broken json')
    if broken == 'nan':
        row['history'][0]['valid_metrics']['ndcg@10'] = float('nan')
        path['result'].write_text(json.dumps(row))
    calls = []
    monkeypatch.setattr(report, 'render', lambda *args: calls.append(args))
    with pytest.raises(ValueError): report.main()
    assert calls == [] and not report.SUMMARY.exists()


@pytest.mark.parametrize('artifact', ('json', 'markdown'))
@pytest.mark.parametrize('write_number', (1, 2))
def test_required_report_write_errors_propagate(reporting, monkeypatch, artifact, write_number):
    reporting.publish()
    original = report.atomic_json if artifact == 'json' else Path.write_text
    attempts = []

    def fail(path, *args, **kwargs):
        attempts.append(path)
        if len(attempts) == write_number:
            raise OSError('required report write failed')
        return original(path, *args, **kwargs)

    if artifact == 'json': monkeypatch.setattr(report, 'atomic_json', fail)
    else: monkeypatch.setattr(Path, 'write_text', fail)
    with pytest.raises(OSError, match='required report write failed'): report.main()


def test_pipeline_success_survives_only_plot_error(reporting, monkeypatch):
    from experiments.mamba3_context_time import pipeline
    monkeypatch.setattr(pipeline, 'RUNTIME', reporting.root / 'pipeline')
    monkeypatch.setattr(pipeline, 'HERE', reporting.root)
    monkeypatch.setattr(pipeline, 'GPU_EVIDENCE', reporting.root / 'runs/gpu.json')
    monkeypatch.setattr(pipeline, 'SUMMARY', report.SUMMARY)
    monkeypatch.setattr(pipeline, 'paths', reporting.paths)
    monkeypatch.setattr(pipeline, 'require_submission', lambda: dict(source_hash='fixture'))
    monkeypatch.setattr(pipeline.signal, 'signal', lambda *args: None)
    monkeypatch.setenv('SLURM_JOB_ID', 'report-fixture')
    monkeypatch.setenv('RUN_COMMIT', 'a' * 40)

    def fail(*args):
        raise RuntimeError('synthetic plot failure')

    monkeypatch.setattr(report, 'render', fail)
    calls = []

    def child(module, args, stage_dir, deadline):
        stage = args[-1] if args else module.rsplit('.', 1)[-1]
        calls.append(stage)
        if stage in reporting.rows: reporting.publish(stage)
        elif stage == 'report': report.main()

    monkeypatch.setattr(pipeline, 'run_child', child)
    pipeline.main()
    status = json.loads((reporting.root / 'pipeline/pipeline_status.json').read_text())
    assert status['status'] == 'PASS'
    assert json.loads(report.SUMMARY.read_text())['plot_status'] == 'FAILED'
    assert calls == ['preflight', 'gpu_checks'] + [t['mode'] for t in plan()['tasks']] + ['report']
