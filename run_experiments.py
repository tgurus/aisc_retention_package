"""Run all experiments for AISC paper v4."""
import sys, os, csv, time
import numpy as np
import scipy.sparse.linalg as spla
from collections import defaultdict

sys.path.insert(0, '/home/claude/aisc_package')
from aisc_retention.meshgen import generate_mesh_grid
from aisc_retention.descriptors import (compute_all, adjacency_matrix,
                                          normalized_laplacian, cotan_laplacian)
from aisc_retention.operators import (radial_clip, decimate_random,
                                        decimate_edge_collapse)
from aisc_retention.utils import median_edge_length

OUTDIR = '/home/claude/aisc_data'
os.makedirs(OUTDIR, exist_ok=True)


def _write(path, rows, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"  → {path} ({len(rows)} rows)")


def _ret(v, b):
    if v is not None and b is not None and b != 0:
        return round(v / b, 6)
    return None


def _delta(rl, rm):
    if rl is not None and rm is not None:
        return round(abs(rl - rm), 6)
    return None


def main():
    print("="*70)
    print("GENERATING MESH GRID")
    print("="*70)
    t0 = time.time()
    meshes = generate_mesh_grid()
    print(f"{len(meshes)} meshes generated in {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════
    # BASELINES
    # ══════════════════════════════════════════════
    print("\n" + "="*70)
    print("EXPT 1: BASELINES")
    print("="*70)
    t0 = time.time()
    fields = ['mesh_type','param_name','param_value','size_target','replicate_id','random_seed',
              'n_vertices_before','n_faces_before',
              'n_lambda2_graph','n_mu2_cotan','H_degree','H_curvature',
              'fiedler_ipr','conductance','valid_mesh_flag']
    rows = []
    baselines = {}
    for mtype, pname, pval, mesh in meshes:
        d = compute_all(mesh)
        r = {
            'mesh_type': mtype, 'param_name': pname, 'param_value': pval,
            'size_target': f'{len(mesh.vertices)}v', 'replicate_id': 0, 'random_seed': 0,
            'n_vertices_before': d['n_vertices'], 'n_faces_before': d['n_faces'],
            'n_lambda2_graph': d['n_lambda2_graph'], 'n_mu2_cotan': d['n_mu2_cotan'],
            'H_degree': d['H_degree'], 'H_curvature': d['H_curvature'],
            'fiedler_ipr': d['fiedler_ipr'], 'conductance': d['conductance'],
            'valid_mesh_flag': d['valid_mesh_flag'],
        }
        rows.append(r)
        baselines[(mtype, pname, pval)] = r
        print(f"  {mtype:18s} {pname}={pval}: λ₂={d['n_lambda2_graph']:.3f}, μ₂={d['n_mu2_cotan']:.2f}")
    _write(f'{OUTDIR}/baselines.csv', rows, fields)
    print(f"  {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════
    # RADIAL CLIPPING
    # ══════════════════════════════════════════════
    print("\n" + "="*70)
    print("EXPT 2: RADIAL CLIPPING")
    print("="*70)
    t0 = time.time()
    fields = ['mesh_type','param_name','param_value','size_target','replicate_id','random_seed',
              'clipping_pct','n_vertices_before','n_faces_before','n_vertices_after','n_faces_after',
              'n_components','largest_component_fraction','valid_mesh_flag',
              'baseline_n_lambda2_graph','baseline_n_mu2_cotan',
              'baseline_H_degree','baseline_H_curvature',
              'n_lambda2_graph','n_mu2_cotan','H_degree','H_curvature',
              'retention_lambda2_graph','retention_mu2_cotan',
              'retention_H_degree','retention_H_curvature','delta_lambda2_mu2']
    rows = []
    clip_levels = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    for mtype, pname, pval, mesh in meshes:
        bl = baselines[(mtype, pname, pval)]
        for pct in clip_levels:
            clipped, nc, lcf = radial_clip(mesh, pct)
            if len(clipped.vertices) < 10 or len(clipped.faces) < 5:
                continue
            d = compute_all(clipped)
            rl = _ret(d['n_lambda2_graph'], bl['n_lambda2_graph'])
            rm = _ret(d['n_mu2_cotan'], bl['n_mu2_cotan'])
            rhd = _ret(d['H_degree'], bl['H_degree'])
            rhc = _ret(d['H_curvature'], bl['H_curvature'])
            rows.append({
                'mesh_type': mtype, 'param_name': pname, 'param_value': pval,
                'size_target': f'{len(mesh.vertices)}v', 'replicate_id': 0, 'random_seed': 0,
                'clipping_pct': pct,
                'n_vertices_before': len(mesh.vertices), 'n_faces_before': len(mesh.faces),
                'n_vertices_after': d['n_vertices'], 'n_faces_after': d['n_faces'],
                'n_components': nc, 'largest_component_fraction': round(lcf, 4),
                'valid_mesh_flag': d['valid_mesh_flag'],
                'baseline_n_lambda2_graph': bl['n_lambda2_graph'],
                'baseline_n_mu2_cotan': bl['n_mu2_cotan'],
                'baseline_H_degree': bl['H_degree'], 'baseline_H_curvature': bl['H_curvature'],
                'n_lambda2_graph': d['n_lambda2_graph'], 'n_mu2_cotan': d['n_mu2_cotan'],
                'H_degree': d['H_degree'], 'H_curvature': d['H_curvature'],
                'retention_lambda2_graph': rl, 'retention_mu2_cotan': rm,
                'retention_H_degree': rhd, 'retention_H_curvature': rhc,
                'delta_lambda2_mu2': 0.0 if pct == 0 else _delta(rl, rm),
            })
        print(f"  {mtype} {pname}={pval}: done ({time.time()-t0:.1f}s)")
    _write(f'{OUTDIR}/radial_clipping_curves.csv', rows, fields)

    # ══════════════════════════════════════════════
    # DECIMATION v2 (LCC cleanup) — RANDOM SUBSAMPLE
    # ══════════════════════════════════════════════
    print("\n" + "="*70)
    print("EXPT 3: DECIMATION v2 (RANDOM SUBSAMPLE + LCC)")
    print("="*70)
    t0 = time.time()
    fields = ['mesh_type','param_name','param_value','size_target','replicate_id','random_seed',
              'retention_pct','n_vertices_original','n_faces_original',
              'n_verts_post_dec','n_faces_post_dec','n_components_pre_lcc','lcc_fraction_of_dec',
              'n_vertices_lcc','n_faces_lcc','valid_mesh_flag',
              'baseline_n_lambda2_graph','baseline_n_mu2_cotan',
              'baseline_H_degree','baseline_H_curvature',
              'n_lambda2_graph','n_mu2_cotan','H_degree','H_curvature',
              'retention_lambda2_graph','retention_mu2_cotan',
              'retention_H_degree','retention_H_curvature','delta_lambda2_mu2']
    rows = []
    ret_levels = [100, 90, 80, 70, 60, 50, 40, 30, 20]
    n_reps = 3

    for mtype, pname, pval, mesh in meshes:
        bl = baselines[(mtype, pname, pval)]
        for rp in ret_levels:
            for rep in range(n_reps):
                seed = rep * 42 + 7
                if rp == 100:
                    d = compute_all(mesh)
                    pre = {'n_verts_post_dec': len(mesh.vertices), 'n_faces_post_dec': len(mesh.faces),
                           'n_components_pre_lcc': 1, 'lcc_fraction_of_dec': 1.0}
                    dv, df = len(mesh.vertices), len(mesh.faces)
                else:
                    lcc, pre = decimate_random(mesh, rp, seed)
                    if lcc is None:
                        rows.append({
                            'mesh_type': mtype, 'param_name': pname, 'param_value': pval,
                            'size_target': f'{len(mesh.vertices)}v',
                            'replicate_id': rep, 'random_seed': seed, 'retention_pct': rp,
                            'n_vertices_original': len(mesh.vertices), 'n_faces_original': len(mesh.faces),
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
                        continue
                    d = compute_all(lcc)
                    dv, df = len(lcc.vertices), len(lcc.faces)
                rl = _ret(d['n_lambda2_graph'], bl['n_lambda2_graph'])
                rm = _ret(d['n_mu2_cotan'], bl['n_mu2_cotan'])
                rows.append({
                    'mesh_type': mtype, 'param_name': pname, 'param_value': pval,
                    'size_target': f'{len(mesh.vertices)}v',
                    'replicate_id': rep, 'random_seed': seed, 'retention_pct': rp,
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
        print(f"  {mtype} {pname}={pval}: done ({time.time()-t0:.1f}s)")
    _write(f'{OUTDIR}/decimation_curves_v2.csv', rows, fields)

    # ══════════════════════════════════════════════
    # NEW: EDGE-COLLAPSE DECIMATION
    # ══════════════════════════════════════════════
    print("\n" + "="*70)
    print("EXPT 4: EDGE-COLLAPSE DECIMATION (topology-preserving)")
    print("="*70)
    t0 = time.time()
    fields_ec = ['mesh_type','param_name','param_value','size_target','replicate_id','random_seed',
                 'retention_pct','simplifier',
                 'n_vertices_original','n_faces_original',
                 'n_verts_post_dec','n_faces_post_dec','n_components_pre_lcc','lcc_fraction_of_dec',
                 'n_vertices_lcc','n_faces_lcc','valid_mesh_flag',
                 'baseline_n_lambda2_graph','baseline_n_mu2_cotan',
                 'baseline_H_degree','baseline_H_curvature',
                 'n_lambda2_graph','n_mu2_cotan','H_degree','H_curvature',
                 'retention_lambda2_graph','retention_mu2_cotan',
                 'retention_H_degree','retention_H_curvature','delta_lambda2_mu2']
    rows_ec = []
    # Use fewer retention levels and 1 replicate since edge-collapse is deterministic under fixed seed
    ret_levels_ec = [100, 90, 80, 70, 60, 50, 40]
    # To save time, run on smaller meshes in each family (the small variant)
    # But use the full 15-mesh grid for completeness
    for mtype, pname, pval, mesh in meshes:
        bl = baselines[(mtype, pname, pval)]
        for rp in ret_levels_ec:
            seed = 7  # deterministic
            if rp == 100:
                d = compute_all(mesh)
                pre = {'n_verts_post_dec': len(mesh.vertices), 'n_faces_post_dec': len(mesh.faces),
                       'n_components_pre_lcc': 1, 'lcc_fraction_of_dec': 1.0}
                dv, df = len(mesh.vertices), len(mesh.faces)
            else:
                lcc, pre = decimate_edge_collapse(mesh, rp, seed)
                if lcc is None:
                    rows_ec.append({
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
                    continue
                d = compute_all(lcc)
                dv, df = len(lcc.vertices), len(lcc.faces)
            rl = _ret(d['n_lambda2_graph'], bl['n_lambda2_graph'])
            rm = _ret(d['n_mu2_cotan'], bl['n_mu2_cotan'])
            rows_ec.append({
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
        print(f"  {mtype} {pname}={pval}: done ({time.time()-t0:.1f}s)")
    _write(f'{OUTDIR}/decimation_edge_collapse.csv', rows_ec, fields_ec)

    # ══════════════════════════════════════════════
    # SPECTRAL TRUNCATION
    # ══════════════════════════════════════════════
    print("\n" + "="*70)
    print("EXPT 5: SPECTRAL TRUNCATION")
    print("="*70)
    t0 = time.time()
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
    _write(f'{OUTDIR}/spectral_truncation.csv', rows, fields)

    # ══════════════════════════════════════════════
    # DISAGREEMENT SUMMARY
    # ══════════════════════════════════════════════
    print("\n" + "="*70)
    print("EXPT 6: DISAGREEMENT SUMMARY")
    print("="*70)
    fields = ['experiment','mesh_type','param_value','transform_level',
              'mean_delta_lambda2_mu2','max_delta_lambda2_mu2','n_valid']
    rows = []
    sources = [
        ('radial_clipping_curves.csv', 'radial_clipping', 'clipping_pct'),
        ('decimation_curves_v2.csv', 'decimation', 'retention_pct'),
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
    _write(f'{OUTDIR}/disagreement_summary.csv', rows, fields)

    print("\n" + "="*70)
    print("ALL SYNTHETIC EXPERIMENTS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
