"""Publication-only regression tests; never import experiment/model modules."""

import csv
import io
import re
import statistics
import subprocess
import unittest
from unittest.mock import patch
from urllib.parse import unquote
from xml.etree import ElementTree

import report


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = report.records()
        cls.summary = report.aggregate(cls.rows)

    def test_exact_source_index_and_roles(self):
        sources = report.sources(self.rows)['sources']
        self.assertEqual(len(sources), 10)
        self.assertEqual([s['role'] for s in sources].count('pilot'), 2)
        self.assertEqual([s['role'] for s in sources].count('parent'), 1)
        self.assertEqual([s['role'] for s in sources].count('new'), 7)
        parent = next(s for s in sources if s['role'] == 'parent')
        self.assertEqual(parent['job_id'], '4355052')
        self.assertNotIn('attempt_003', parent['path'])

    def test_reject_duplicate_source(self):
        entries = report.read(report.BASE / 'resume_plan_003.json')['source_index']
        with self.assertRaisesRegex(ValueError, 'duplicated'):
            report.select_sources(entries + [entries[0]])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            report.select_sources(entries[:-1])

    def test_reject_failed_run_and_test_usage(self):
        original = report.read
        first = self.rows[0]['entry']['path']
        for field, value in [('status', 'FAIL'), ('test_evaluation_count', 1)]:
            def modified(path):
                data = original(path)
                if str(path) == first:
                    data[field] = value
                return data
            with self.subTest(field=field), patch.object(report, 'read', side_effect=modified):
                with self.assertRaises(ValueError):
                    report.records()

    def test_five_pairs_and_negative_seed(self):
        pairs = self.summary['pairs']
        self.assertEqual([(p['dual'], p['triple']) for p in pairs],
                         [(.0615, .0623), (.0632, .0628), (.0627, .0630), (.0625, .0635), (.0627, .0633)])
        self.assertAlmostEqual(pairs[1]['delta'], -.0004)
        self.assertEqual(self.summary['new_four_pairs']['full']['negative'], 1)
        self.assertEqual(self.summary['all_five_pairs']['full']['positive'], 4)

    def test_arithmetic_four_and_five(self):
        for name, values in [('new_four_pairs', (.062775, .063150, .000375)),
                             ('all_five_pairs', (.062520, .062980, .000460))]:
            stats = self.summary[name]['full']
            for mode, expected in zip(('dual', 'triple', 'delta'), values):
                self.assertAlmostEqual(stats[mode]['mean'], expected, places=14)
            self.assertAlmostEqual(stats['relative_percent'], 100 * values[2] / values[0])

    def test_sample_std_not_population_or_difference_of_stds(self):
        pairs = self.summary['pairs'][1:]
        stats = self.summary['new_four_pairs']['full']
        differences = [p['delta'] for p in pairs]
        self.assertEqual(stats['delta']['sample_std'], statistics.stdev(differences))
        self.assertNotEqual(stats['delta']['sample_std'], statistics.pstdev(differences))
        self.assertNotEqual(stats['delta']['sample_std'], stats['triple']['sample_std'] - stats['dual']['sample_std'])

    def test_first27_not_full_horizon(self):
        pairs = self.summary['pairs']
        self.assertEqual(pairs[1]['dual_first27'], .0614)
        self.assertEqual(pairs[1]['dual'], .0632)
        self.assertLess(pairs[-1]['triple_first27'] - pairs[-1]['dual_first27'], 0)
        self.assertAlmostEqual(self.summary['new_four_pairs']['first27']['delta']['mean'], .000450)
        self.assertAlmostEqual(self.summary['all_five_pairs']['first27']['delta']['mean'], .000520)

    def test_rounding_and_tex_values(self):
        s = self.summary['new_four_pairs']['full']
        self.assertEqual(report.pm(s['dual']), '0.062775 ± 0.000299')
        text = report.tex(self.summary)
        self.assertIn('2027 & 0.0632 & 0.0628 & $-0.0004$', text)
        self.assertIn(r'0.000375 \pm 0.000591', text)
        self.assertIn(r'0.000460 \pm 0.000546', text)
        self.assertIn('not a TEST benchmark', text)
        self.assertNotIn(r'{|', text)

    def test_svg_points_and_zero(self):
        root = ElementTree.fromstring(report.svg(self.summary))
        ns = {'s': 'http://www.w3.org/2000/svg'}
        dots = root.findall('.//s:circle', ns)
        self.assertEqual(len(dots), 5)
        self.assertEqual(dots[0].attrib['fill'], 'white')
        self.assertLess(float(dots[1].attrib['cx']), 540)
        self.assertTrue(any(l.attrib.get('x1') == '540.0' and l.attrib.get('x2') == '540.0'
                            for l in root.findall('.//s:line', ns)))
        self.assertEqual(len(root.findall('.//s:polyline', ns)), 0)

    def test_generated_files_and_tables(self):
        for name, content in report.outputs(self.rows, self.summary).items():
            self.assertEqual((report.HERE / name).read_text(), content, name)
        text = (report.ROOT / report.REPORT).read_text()
        for name, content in report.tables(self.summary).items():
            self.assertIn(f'<!-- siso:{name}:start -->\n{content}\n<!-- siso:{name}:end -->', text)

    def test_registry_exact_rows_and_prefix(self):
        result = report.registry(self.rows)
        self.assertEqual((result['before'], result['added'], result['after']), (68, 10, 78))
        rows = list(csv.DictReader(io.StringIO((report.ROOT / 'experiments/results.csv').read_text())))
        new = [r for r in rows if r['run_id'] in {x['raw']['run_id'] for x in self.rows}]
        self.assertEqual(len(new), 10)
        self.assertTrue(all(r['test_evaluation_count'] == '0' and r['split'] == 'validation' for r in new))

    def test_original_report_sections_and_test_tables_unchanged(self):
        def old(path):
            return subprocess.check_output(['git', 'show', f'{report.OLD_MAIN}:{path}'], cwd=report.ROOT).decode()
        marker = '<a id="context-time-pilot"></a>'
        self.assertEqual((report.ROOT / report.REPORT).read_text().split(marker, 1)[1],
                         old(report.REPORT).split(marker, 1)[1])
        path = 'reports/PAPER_RESULTS.md'
        table_lines = lambda t: [x for x in t.splitlines() if x.startswith('|')]
        self.assertEqual(table_lines((report.ROOT / path).read_text()), table_lines(old(path)))

    def test_relative_links_and_explicit_anchors(self):
        for path in (report.ROOT / report.REPORT, report.HERE / 'PUBLICATION_AUDIT.md'):
            text = path.read_text()
            for link in re.findall(r'\]\(([^)]+)\)', text):
                if '://' in link:
                    continue
                target, _, anchor = unquote(link).partition('#')
                destination = (path.parent / target).resolve() if target else path
                self.assertTrue(destination.is_file(), str(destination))
                if anchor:
                    self.assertIn(f'id="{anchor}"', destination.read_text())
        for path in ('README.md', 'reports/RESULTS.md', 'reports/PAPER_RESULTS.md'):
            self.assertIn('#' + report.ANCHOR, (report.ROOT / path).read_text())

    def test_diagnostics_are_best_epoch_and_head_maxima(self):
        for d in self.summary['best_triple_diagnostics']:
            raw = next(r['raw'] for r in self.rows if r['raw']['seed'] == d['seed'] and r['raw']['mode'] == 'triple')
            best = raw['history'][raw['best_epoch']]['diagnostics']
            self.assertEqual(d['max_head_write_below_bound'], max(h[0] for h in best['near_bound_fractions'][1]))
            self.assertEqual(d['max_head_phase_above_bound'], max(h[1] for h in best['near_bound_fractions'][2]))


if __name__ == '__main__':
    unittest.main()
