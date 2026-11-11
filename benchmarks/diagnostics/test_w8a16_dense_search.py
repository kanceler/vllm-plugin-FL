import json
from pathlib import Path
import tempfile
import unittest
from w8a16_dense_search import SHAPES, WIDE_MS, REFINED_MS, CONTROLS, generate_candidates, pool_from_wide, shortlist
from tune_metax_w8a16_dense import aggregate, write_json

class DenseTests(unittest.TestCase):
    def test_matrix_and_domain(self):
        self.assertEqual(sum(map(len,WIDE_MS.values())),6)
        self.assertEqual(sum(map(len,REFINED_MS.values())),10)
        for shape in SHAPES:
            self.assertFalse(set(WIDE_MS[shape]) & set(REFINED_MS[shape]))
            rows=generate_candidates(24,*SHAPES[shape],104,64,256*1024**2)
            self.assertEqual(len(rows),576)
            kept={tuple(r['config']) for r in rows if r['status']=='scheduled'}
            self.assertTrue(set(CONTROLS)<=kept)
            self.assertIn('pruned_heuristic',{r['status'] for r in rows})
    def test_rejected_config_not_propagated(self):
        c=[16,256,32,2,1]
        p={'m':24,'screening':[{'config':c,'status':'passed','median_ms':1}],
           'independent_rejections':[{'config':c}]}
        self.assertNotIn(c,[r['config'] for r in pool_from_wide([p])])
    def test_dense_shortlist_uses_five_fields(self):
        rows=[{'config':[16,64,128,4,2],'status':'passed','median_ms':1.0},
              {'config':[32,64,128,4,2],'status':'passed','median_ms':1.005}]
        self.assertEqual(len(shortlist(rows,top_k=1)),2)
    def test_no_winner_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for shape in SHAPES:
                for m in (*WIDE_MS[shape],*REFINED_MS[shape]):
                    p=root/shape/f'm{m}'
                    p.mkdir(parents=True)
                    write_json(p/'result.json',{'stage':'wide','complete':False,
                        'selected_config':None,'results':[],
                        'beats_bf16_over_1pct_on_both_seeds':False})
            result=aggregate(root)
            self.assertTrue(result['execution_complete'])
            self.assertFalse(result['complete'])
            self.assertEqual(len(result['failed_points']),16)
