"""Publication guards; no writes to raw evidence and no model work."""
import hashlib
import importlib.util
import json
import re
import subprocess
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('headtime_publish',HERE/'publish.py')
publish=importlib.util.module_from_spec(spec);spec.loader.exec_module(publish)


class PublicationTests(unittest.TestCase):
    def test_pilot_numbers_come_from_records(self):
        rows=publish.pilot_records();section=publish.pilot_section(rows)
        self.assertEqual([r['best_valid_score'] for r in rows],[.0633,.0627,.0639])
        self.assertIn('+0.0012',section);self.assertIn('-0.0006',section)
        self.assertIn('0.0620',section);self.assertIn('0.0617',section);self.assertIn('0.0621',section)
        self.assertIn('не доказанный период',section)

    def test_original_csv_prefix_and_test_publication(self):
        root=publish.ROOT
        old=subprocess.check_output(['git','show',publish.OLD_MAIN+':experiments/results.csv'],cwd=root)
        self.assertTrue((root/'experiments/results.csv').read_bytes().startswith(old))
        old_test=subprocess.check_output(['git','show',publish.OLD_MAIN+':reports/PAPER_RESULTS.md'],cwd=root)
        self.assertEqual((root/'reports/PAPER_RESULTS.md').read_bytes(),old_test)

    def test_pilot_raw_matches_preservation(self):
        p=publish.PACKAGE
        manifest=publish.read(p/'evidence/job4362620/preservation_manifest.json')
        for name,row in manifest['files'].items():
            archive=p/'evidence/job4362620/files'/Path(name).relative_to(p.relative_to(publish.ROOT))
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(),row['sha256'])
            if '/runs/attempt_002/' in name:self.assertEqual((publish.ROOT/name).read_bytes(),archive.read_bytes())

    def test_numeric_summary_and_paired_table_are_complete_if_published(self):
        path=HERE/'confirmation_summary.json'
        if not path.exists():
            self.assertNotIn('<a id="head-timescales-confirmation"></a>',publish.REPORT.read_text())
            return
        summary=publish.read(path)
        self.assertEqual(summary['scientific_fits_completed'],12)
        self.assertEqual(summary['complete_new_triples'],4)
        self.assertEqual(len(summary['rows']),15)
        for key,n in [('new4_full',4),('all5_full',5)]:
            for d in summary['cohorts'][key]['contrasts'].values():
                self.assertEqual(d['n_available'],n)
                self.assertEqual(d['positive']+d['zero']+d['negative'],n)
        self.assertIn(publish.confirmation_section(summary),publish.REPORT.read_text())
        tex=publish.confirmation_tex(summary)
        self.assertEqual(tex,(HERE/'valid_table.tex').read_text())
        self.assertEqual(tex.count('{'),tex.count('}'))
        self.assertNotIn('All five',tex)
        self.assertEqual(tex.count(' & 4 & '),3)
        self.assertIn('не выбираются как обязательное усложнение',publish.REPORT.read_text())
        self.assertIn('+ / − / 0',publish.REPORT.read_text())
        import xml.etree.ElementTree as ET
        ET.parse(HERE/'paired_delta.svg')

    def test_published_source_links(self):
        text=publish.REPORT.read_text().split('<a id="head-timescales-pilot"></a>')[1]
        for target in re.findall(r'\]\(([^)]+)\)',text):
            path,_,anchor=target.partition('#')
            if '://' in path:continue
            dest=(publish.REPORT.parent/path).resolve() if path else publish.REPORT
            self.assertTrue(dest.exists(),target)
            if anchor:self.assertIn('id="'+anchor+'"',dest.read_text())


if __name__=='__main__':unittest.main()
