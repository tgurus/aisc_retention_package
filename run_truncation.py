"""Spectral truncation + disagreement summary experiments."""
import sys, os, csv, time
import numpy as np
import scipy.sparse.linalg as spla
from collections import defaultdict
sys.path.insert(0, '/home/claude/aisc_package')

from aisc_retention.meshgen import generate_mesh_grid
from aisc_retention.descriptors import (adjacency_matrix, normalized_laplacian,
                                          cotan_laplacian)
from aisc_retention.utils import median_edge_length

OUTDIR = '/home/claude/aisc_data'

# Spectral truncation
print("="*70)
print("SPECTRAL TRUNCATION")
print("="*70)
t0 = time.time()
meshes = generate_mesh_grid()
fields = ['mesh_type','param_name','param_value','size_target','replicate_id','random_seed',
          'operator_type','k_modes','n_vertices_before','n_faces_before',
          'spectral_entropy_k','heat_trace_t1_k','heat_trace_t10_k',
          'lambda2_or_mu2','valid_mesh_flag']
k_values = [2, 5, 10, 20, 50]
rows = []
for mtype, pname, pval, mesh in meshes:
    n = len(mesh.vertices)
    h = median_edge_length(mesh)
    t1, t10 = 0.5 * h**2, 50.0 * h**2
    k_max = min(50, n - 2)
    W = adjacency_matrix(mesh); Lsym = normalized_laplacian(W)
    evals_g, _ = spla.eigsh(Lsym, k=k_max, sigma=1e-8, which='LM')
    evals_g = np.sort(evals_g)
    C, M = cotan_laplacian(mesh)
    evals_c, _ = spla.eigsh(C, M=M, k=k_max, sigma=1e-8, which='LM')
    evals_c = np.sort(evals_c)
    for k in k_values:
        if k > len(evals_g): continue
        for op, ev in [('graph', evals_g), ('cotan', evals_c)]:
            ek = ev[:k]
            ht1 = float(np.sum(np.exp(-t1 * ek)))
            ht10 = float(np.sum(np.exp(-t10 * ek)))
            t_mid = np.sqrt(t1 * t10)
            p = np.exp(-t_mid * ek); p = p / (p.sum() + 1e-15)
            se = float(-np.sum(p * np.log2(p + 1e-15)))
            rows.append({
                'mesh_type': mtype, 'param_name': pname, 'param_value': pval,
                'size_target': f'{n}v', 'replicate_id': 0, 'random_seed': 0,
                'operator_type': op, 'k_modes': k,
                'n_vertices_before': n, 'n_faces_before': len(mesh.faces),
                'spectral_entropy_k': round(se, 8),
                'heat_trace_t1_k': round(ht1, 8),
                'heat_trace_t10_k': round(ht10, 8),
                'lambda2_or_mu2': float(ek[1]) if len(ek) > 1 else None,
                'valid_mesh_flag': 1,
            })
    print(f"  {mtype} {pname}={pval}: done ({time.time()-t0:.1f}s)")

with open(f'{OUTDIR}/spectral_truncation.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f"  → spectral_truncation.csv ({len(rows)} rows)")


# Disagreement summary
print("\n" + "="*70)
print("DISAGREEMENT SUMMARY")
print("="*70)
fields = ['experiment','mesh_type','param_value','transform_level',
          'mean_delta_lambda2_mu2','max_delta_lambda2_mu2','n_valid']
rows = []
sources = [
    ('radial_clipping_curves.csv', 'radial_clipping', 'clipping_pct'),
    ('decimation_curves_v2.csv', 'decimation_random', 'retention_pct'),
    ('decimation_edge_collapse.csv', 'decimation_edge_collapse', 'retention_pct'),
]
for fname, exp_name, level_col in sources:
    path = f'{OUTDIR}/{fname}'
    if not os.path.exists(path): continue
    with open(path) as f:
        data = list(csv.DictReader(f))
    groups = defaultdict(list)
    for r in data:
        d = r.get('delta_lambda2_mu2', '')
        if d and d not in ('None','','0.0'):
            key = (r['mesh_type'], r.get('param_value',''), r[level_col])
            groups[key].append(float(d))
    for (mt, pv, lv), vals in sorted(groups.items()):
        rows.append({
            'experiment': exp_name, 'mesh_type': mt, 'param_value': pv,
            'transform_level': lv,
            'mean_delta_lambda2_mu2': round(np.mean(vals), 6),
            'max_delta_lambda2_mu2': round(np.max(vals), 6),
            'n_valid': len(vals),
        })

with open(f'{OUTDIR}/disagreement_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f"  → disagreement_summary.csv ({len(rows)} rows)")
