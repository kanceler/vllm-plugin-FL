# SPDX-License-Identifier: Apache-2.0
"""Finite Dense pool for the approved Gate/Up and O middle-M experiment."""
from itertools import product
from collections import Counter
from w8a16_splitk_search import config_id

SHAPES = {'gate_up': (12288, 2048), 'o': (2048, 2048)}
WIDE_MS = {'gate_up': (24, 64, 96), 'o': (64, 80, 96)}
REFINED_MS = {'gate_up': (16, 32, 48, 80, 104), 'o': (56, 72, 88, 104, 112)}
POLICY = {'version': 1, 'pipeline': 'basic', 'reduce_block': 256,
          'top_k': 5, 'near_tie_ratio': 1.01, 'control_stability_threshold_pct': 2.0,
          'scope': 'finite Dense pool; 6 wide + 10 refined; O M56 boundary probe'}
CONTROLS = ((16,64,256,4,3), (32,32,128,4,3), (32,64,128,4,3),
            (64,64,128,4,3), (64,128,64,8,3), (32,128,64,2,3),
            (16,64,128,4,2), (32,64,128,4,2), (128,128,32,8,2))


def legality(n, k, config):
    if len(config) != 5 or any(x <= 0 for x in config):
        return 'invalid_config'
    return None


def generate_candidates(m, n, k, sm_count, warp_size, workspace_budget):
    rows = []
    domain = set(product((16,32,64,128), (32,64,128,256),
                         (32,64,128,256), (2,4,8), (1,2,3))) | set(CONTROLS)
    for c in sorted(domain):
        bm,bn,bk,w,st=c
        reasons = ['excess_M_padding'] if bm > max(16, 2*m) else []
        sources = ['finite_dense_pool']
        if c in CONTROLS: sources.append('historical_control')
        if reasons and w==4 and st==2: sources.append('heuristic_audit')
        keep=not reasons or c in CONTROLS or 'heuristic_audit' in sources
        tiles=((m+bm-1)//bm)*((n+bn-1)//bn)
        rows.append({'config':list(c),'config_id':config_id(c), 'sources':sources,
                     'heuristic_reasons':reasons,
                     'status':'scheduled' if keep else 'pruned_heuristic',
                     'features':{'output_programs':tiles,'programs_per_sm':tiles/sm_count,
                                 'k_loops':(k+bk-1)//bk,'workspace_bytes':0,
                                 'eta_mn':m*n/(tiles*bm*bn)}})
    return rows


def shortlist(measurements, top_k=5, near_tie_ratio=1.01):
    passed=sorted((r for r in measurements if r['status']=='passed'),
                  key=lambda r:(r['median_ms'],tuple(r['config'])))
    if not passed:return []
    selected={tuple(r['config']):r for r in passed[:top_k]}
    structures={tuple(r['config'][:3]) for r in selected.values()}
    for r in passed:
        if r['median_ms'] > passed[0]['median_ms']*near_tie_ratio:break
        structure=tuple(r['config'][:3])
        if structure not in structures:
            selected[tuple(r['config'])]=r
            structures.add(structure)
    return list(selected.values())


def pool_from_wide(points):
    pool={c:{'historical_control'} for c in CONTROLS}
    for p in points:
        rejected={tuple(r['config']) for r in p.get('independent_rejections',[])}
        for r in shortlist([r for r in p['screening'] if tuple(r['config']) not in rejected]):
            pool.setdefault(tuple(r['config']),set()).add(f"wide_M={p['m']}")
    return [{'config':list(c),'sources':sorted(s)} for c,s in sorted(pool.items())]


def plan(sm_count, warp_size, workspace_budget):
    points=[]
    for shape,ms in WIDE_MS.items():
        for m in ms:
            rows=generate_candidates(m,*SHAPES[shape],sm_count,warp_size,workspace_budget)
            points.append({'shape':shape,'m':m,'counts':dict(Counter(r['status'] for r in rows)),
                           'candidates':rows})
    return {'policy':POLICY,'wide_points':points,'refined_ms':REFINED_MS,
            'hardware_inputs':{'sm_count':sm_count,'warp_size':warp_size}}
