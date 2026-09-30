"""Reporting checks only: no experiment imports, model work or checkpoint loading."""
import hashlib
import math
import re
import subprocess
import unittest
import xml.etree.ElementTree as ET

import report


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.sources = report.records()
        cls.summary = report.aggregate(cls.rows)

    def test_source_selection_and_metadata(self):
        entries = self.sources['sources']
        self.assertEqual(len(entries), 25)
        self.assertEqual(len({e['path'] for e in entries}), 25)
        self.assertEqual(sum(e['role'] == 'pilot' for e in entries), 5)
        self.assertEqual(sum(e['architecture'] == 'SISO' for e in entries), 10)
        parent = next(e for e in entries if e['architecture'] == 'SISO' and e['seed'] == 2027 and e['mode'] == 'dual')
        self.assertEqual(parent['job_id'], '4355052')
        for e in entries:
            self.assertEqual(report.sha(e['path']), e['sha256'])
            self.assertEqual(report.sha(e['metadata_path']), e['metadata_sha256'])
            self.assertEqual(e['split'], 'VALID')

    def test_sample_std_and_last_tie(self):
        self.assertEqual(report.stats([1, 2, 3])['sample_std'], 1)
        self.assertEqual(report.last_best([{'valid_ndcg10': v} for v in [.1, .2, .1, .2]]), 3)
        r = next(r for r in self.rows if r['architecture'] == 'MIMO' and r['mode'] == 'dual' and r['seed'] == 2028)
        self.assertEqual(report.last_best(r['history']), 46)

    def test_independent_control_values_and_negative_pairs(self):
        expected = {'SISO_new4': (.062775, .063150, .000375, 3, 1),
                    'SISO_all5': (.062520, .062980, .000460, 4, 1),
                    'MIMO_new4': (.062925, .063325, .000400, 3, 1),
                    'MIMO_all5': (.063000, .063020, .000020, 3, 2)}
        for label, (dual, triple, delta, positive, negative) in expected.items():
            c = self.summary['cohorts'][label]
            self.assertAlmostEqual(c['models']['dual']['mean'], dual)
            self.assertAlmostEqual(c['models']['triple']['mean'], triple)
            d = c['contrasts']['triple-dual']
            self.assertAlmostEqual(d['mean'], delta)
            self.assertEqual((d['positive'], d['negative'], d['zero']), (positive, negative, 0))
            manual_std = math.sqrt(sum((x - delta) ** 2 for x in d['deltas']) / (len(d['deltas']) - 1))
            self.assertAlmostEqual(d['sample_std'], manual_std)
        c = self.summary['cohorts']['MIMO_new4']
        self.assertAlmostEqual(c['models']['base']['mean'], .059125)
        self.assertAlmostEqual(c['contrasts']['dual-base']['mean'], .0038)
        self.assertAlmostEqual(c['contrasts']['triple-base']['mean'], .0042)

    def test_first27_subsets_without_padding(self):
        triple = self.summary['cohorts']['MIMO_new4_first27']
        pair = self.summary['cohorts']['MIMO_dual_triple_new4_first27']
        self.assertEqual(triple['seeds'], [2027, 2028, 2030])
        self.assertEqual(pair['seeds'], [2027, 2028, 2029, 2030])
        self.assertAlmostEqual(pair['models']['dual']['mean'], .0618)
        self.assertAlmostEqual(pair['models']['triple']['mean'], .062475)
        self.assertAlmostEqual(pair['contrasts']['triple-dual']['mean'], .000675)
        short = next(r for r in self.rows if r['architecture'] == 'MIMO' and r['mode'] == 'base' and r['seed'] == 2029)
        self.assertEqual(short['status'], 'PASS')
        self.assertIsNone(report.first27(short))
        siso = self.summary['cohorts']['SISO_new4_first27']
        self.assertEqual(siso['seeds'], [2027, 2028, 2029, 2030])

    def test_registry_preserved_and_idempotent(self):
        p = report.ROOT / 'experiments/results.csv'
        before = p.read_bytes()
        result = report.registry(self.rows, self.sources, append=True)
        self.assertEqual((result['before'], result['added']), (81, 12))
        self.assertGreaterEqual(result['after'], 93)
        self.assertEqual(result['later_appended'], result['after'] - 93)
        self.assertEqual(before, p.read_bytes())
        self.assertEqual(result['historical_sha256'], '9c837008a31c643fa6a1e9e633e41c766a60c48d0db1fb815f28bc6d09aee3fa')

    def test_derived_svg_tex_and_report(self):
        for name, expected in {'sources.json': report.dump(self.sources), 'summary.json': report.dump(self.summary),
                               'mimo_paired_delta.svg': report.svg(self.summary), 'siso_mimo_valid_table.tex': report.tex(self.summary)}.items():
            self.assertEqual((report.HERE / name).read_text(), expected)
        svg = ET.fromstring(report.svg(self.summary))
        ns = {'s': 'http://www.w3.org/2000/svg'}
        self.assertEqual(len(svg.findall('.//s:circle', ns)), 4)
        self.assertEqual(len(svg.findall('.//s:path', ns)), 1)
        self.assertIn('-0.0015', report.svg(self.summary))
        self.assertIn('-0.0010', report.svg(self.summary))
        tex = report.tex(self.summary)
        self.assertEqual(tex.count('\\multicolumn'), 4)
        self.assertEqual(tex.count('Base &'), 2)
        self.assertEqual(tex.count('{'), tex.count('}'))
        text = report.REPORT.read_text()
        for name, table in report.tables(self.rows, self.summary).items():
            self.assertIn(f'<!-- mimo:{name}:start -->\n{table}\n<!-- mimo:{name}:end -->', text)
        for anchor in ('siso-dual-triple-confirmation', 'mimo-time-confirmation', 'mimo-time-pilot'):
            self.assertEqual(text.count(f'<a id="{anchor}"></a>'), 1)
        for target in re.findall(r'\]\(([^)]+)\)', text.split('<a id="mimo-time-pilot">')[0]):
            path, _, anchor = target.partition('#')
            dest = (report.REPORT.parent / path).resolve() if path else report.REPORT
            self.assertTrue(dest.exists(), target)
            if anchor:
                self.assertIn(f'id="{anchor}"', dest.read_text())

    def test_historical_assets_and_test_tables_unchanged(self):
        for name in ('reports/assets/three_time_confirmation/paired_delta.svg',
                     'reports/assets/three_time_confirmation/siso_dual_triple_table.tex'):
            old = subprocess.check_output(['git', 'show', report.OLD_MAIN + ':' + name], cwd=report.ROOT)
            self.assertEqual(hashlib.sha256(old).hexdigest(), report.sha(name))
        for name, section in [('reports/PAPER_RESULTS.md', '## Связанные работы 2026 года'),
                              ('reports/RESULTS.md', '## Завершённые исследования')]:
            old = subprocess.check_output(['git', 'show', report.OLD_MAIN + ':' + name], cwd=report.ROOT).decode()
            new = (report.ROOT / name).read_text()
            old_tables = [line for line in old.split(section)[0].splitlines() if line.startswith('|')]
            new_tables = [line for line in new.split(section)[0].splitlines() if line.startswith('|')]
            self.assertEqual(old_tables, new_tables)


if __name__ == '__main__':
    unittest.main()
