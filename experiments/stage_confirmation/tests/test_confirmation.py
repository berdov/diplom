import unittest
from copy import deepcopy
from experiments.stage_confirmation.run import case,verify_sources
from experiments.target_combination_analysis.common import config
from experiments.stage_confirmation.fairness import replay

class ConfirmationTests(unittest.TestCase):
    def test_only_seed_changes(self):
        seen=set()
        for i in range(6):
            cell,cfg=case(i);seen.add((tuple(cell['active_targets']),cfg['training']['seed']))
            restored=deepcopy(cfg);restored['training']['seed']=2026
            self.assertEqual(restored,config())
        self.assertEqual(len(seen),6)
        self.assertEqual({s for _,s in seen},{2027,2028})
        for i in [-1,6,16]:
            with self.assertRaises(ValueError):case(i)
    def test_pinned_training_code(self):
        self.assertGreater(verify_sources(),50)
    def test_full_fairness_replay(self):
        r=replay();self.assertFalse(r['rerun_required'])
        self.assertEqual([x['epoch'] for x in r['validation_trajectory']],[5,10,15,20,25])
        self.assertTrue(all(x['solution_index']==x['old_selected_solution'] for x in r['validation_trajectory']))
        self.assertEqual(r['ranking_operating_point']['metrics']['NDCG@10'],.0522)

if __name__=='__main__':unittest.main()
