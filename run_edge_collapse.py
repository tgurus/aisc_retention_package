"""Run edge-collapse decimation only.

Since edge-collapse is O(n²) in the naive implementation, we restrict to the
small variant of each morphology to stay within time budget.
"""
import sys, os, csv, time
sys.path.insert(0, '/home/claude/aisc_package')

from aisc_retention.meshgen import (smooth_tube, necked_tube, double_neck_tube,
                                     branching_tube, crista_sheet)
from aisc_retention.descriptors import compute_all
from aisc_retention.operators import decimate_edge_collapse

OUTDIR = '/home/claude/aisc_data'


def _ret(v, b):
    return round(v / b, 6) if v is not None and b is not None and b != 0 else None


def _delta(rl, rm):
    return round(abs(rl - rm), 6) if rl is not None and rm is not None else None


# Build a reduced mesh grid — one representative per morphology class (smallest variant)
# to make edge-collapse tractable within the time budget.
meshes = [
    ('smooth_tube', 'n_z', 30, smooth_tube(n_theta=32, n_z=30)),
    ('necked_tube', 'neck_depth', 0.5, necked_tube(n_theta=32, n_z=60, neck_depth=0.5)),
    ('double_neck_tube', 'n_z', 40, double_neck_tube(n_theta=32, n_z=40)),
    ('branching_tube', 'n_theta', 16, branching_tube(n_theta=16)),
    ('crista_sheet', 'n_theta', 16, crista_sheet(n_theta=16)),
]

print(f"Edge-collapse on {len(meshes)} representative meshes")

# Baselines
baselines = {}
for mtype, pname, pval, m in meshes:
    d = compute_all(m)
    baselines[(mtype, pname, pval)] = d
    print(f"  Baseline {mtype}: n={d['n_vertices']}, λ₂={d['n_lambda2_graph']:.3f}, μ₂={d['n_mu2_cotan']:.2f}")

fields = ['mesh_type','param_name','param_value','size_target','replicate_id','random_seed',
          'retention_pct','simplifier',
          'n_vertices_original','n_faces_original',
          'n_verts_post_dec','n_faces_post_dec','n_components_pre_lcc','lcc_fraction_of_dec',
          'n_vertices_lcc','n_faces_lcc','valid_mesh_flag',
          'baseline_n_lambda2_graph','baseline_n_mu2_cotan',
          'baseline_H_degree','baseline_H_curvature',
          'n_lambda2_graph','n_mu2_cotan','H_degree','H_curvature',
          'retention_lambda2_graph','retention_mu2_cotan',
          'retention_H_degree','retention_H_curvature','delta_lambda2_mu2']
rows = []
ret_levels = [100, 90, 80, 70, 60, 50]  # skip 40% (very slow on larger meshes)

t0 = time.time()
for mtype, pname, pval, mesh in meshes:
    bl = baselines[(mtype, pname, pval)]
    for rp in ret_levels:
        t_iter = time.time()
        seed = 7
        if rp == 100:
            d = compute_all(mesh)
            pre = {'n_verts_post_dec': len(mesh.vertices), 'n_faces_post_dec': len(mesh.faces),
                   'n_components_pre_lcc': 1, 'lcc_fraction_of_dec': 1.0}
            dv, df = len(mesh.vertices), len(mesh.faces)
        else:
            lcc, pre = decimate_edge_collapse(mesh, rp, seed)
            if lcc is None:
                rows.append({
                    'mesh_type': mtype, 'param_name': pname, 'param_value': pval,
                    'size_target': f'{len(mesh.vertices)}v',
                    'replicate_id': 0, 'random_seed': seed, 'retention_pct': rp,
                    'simplifier': 'edge_collapse',
                    'n_vertices_original': len(mesh.vertices),
                    'n_faces_original': len(mesh.faces),
                    **pre, 'n_vertices_lcc': 0, 'n_faces_lcc': 0, 'valid_mesh_flag': 0,
                    'baseline_n_lambda2_graph': bl['n_lambda2_graph'],
                    'baseline_n_mu2_cotan': bl['n_mu2_cotan'],
                    'baseline_H_degree': bl['H_degree'],
                    'baseline_H_curvature': bl['H_curvature'],
                    'n_lambda2_graph': None, 'n_mu2_cotan': None,
                    'H_degree': None, 'H_curvature': None,
                    'retention_lambda2_graph': None, 'retention_mu2_cotan': None,
                    'retention_H_degree': None, 'retention_H_curvature': None,
                    'delta_lambda2_mu2': None,
                })
                print(f"  {mtype} ret={rp}%: FAILED ({time.time()-t_iter:.1f}s)")
                continue
            d = compute_all(lcc)
            dv, df = len(lcc.vertices), len(lcc.faces)
        rl = _ret(d['n_lambda2_graph'], bl['n_lambda2_graph'])
        rm = _ret(d['n_mu2_cotan'], bl['n_mu2_cotan'])
        rows.append({
            'mesh_type': mtype, 'param_name': pname, 'param_value': pval,
            'size_target': f'{len(mesh.vertices)}v',
            'replicate_id': 0, 'random_seed': seed, 'retention_pct': rp,
            'simplifier': 'edge_collapse',
            'n_vertices_original': len(mesh.vertices), 'n_faces_original': len(mesh.faces),
            **pre, 'n_vertices_lcc': dv, 'n_faces_lcc': df,
            'valid_mesh_flag': d['valid_mesh_flag'],
            'baseline_n_lambda2_graph': bl['n_lambda2_graph'],
            'baseline_n_mu2_cotan': bl['n_mu2_cotan'],
            'baseline_H_degree': bl['H_degree'], 'baseline_H_curvature': bl['H_curvature'],
            'n_lambda2_graph': d['n_lambda2_graph'], 'n_mu2_cotan': d['n_mu2_cotan'],
            'H_degree': d['H_degree'], 'H_curvature': d['H_curvature'],
            'retention_lambda2_graph': rl if rp < 100 else 1.0,
            'retention_mu2_cotan': rm if rp < 100 else 1.0,
            'retention_H_degree': _ret(d['H_degree'], bl['H_degree']) if rp < 100 else 1.0,
            'retention_H_curvature': _ret(d['H_curvature'], bl['H_curvature']) if rp < 100 else 1.0,
            'delta_lambda2_mu2': 0.0 if rp == 100 else _delta(rl, rm),
        })
        print(f"  {mtype} ret={rp}%: LCC={pre['lcc_fraction_of_dec']}, n={dv}, Δ={rows[-1]['delta_lambda2_mu2']} ({time.time()-t_iter:.1f}s)")

print(f"\nTotal: {time.time()-t0:.1f}s")

with open(f'{OUTDIR}/decimation_edge_collapse.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f"  → {OUTDIR}/decimation_edge_collapse.csv ({len(rows)} rows)")
