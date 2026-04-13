"""Generate all manuscript figures from CSVs."""
import csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

DATA = '/home/claude/aisc_data'
OUT = '/home/claude/figs'
os.makedirs(OUT, exist_ok=True)


def _load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


TYPES = ['smooth_tube', 'necked_tube', 'double_neck_tube', 'branching_tube', 'crista_sheet']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


# ═══════════════════════════════════════════════════════════
# Figure 1: Baselines
# ═══════════════════════════════════════════════════════════
baselines = _load(f'{DATA}/baselines.csv')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for i, mt in enumerate(TYPES):
    rows = [r for r in baselines if r['mesh_type'] == mt]
    gl2 = [float(r['n_lambda2_graph']) for r in rows]
    mu2 = [float(r['n_mu2_cotan']) for r in rows]
    ax.scatter(gl2, mu2, c=COLORS[i], s=80, edgecolors='k', linewidths=0.5,
               label=mt.replace('_', ' '), zorder=3)
ax.set_xlabel(r'$n \cdot \lambda_2^G$ (graph)', fontsize=11)
ax.set_ylabel(r'$n \cdot \mu_2^C$ (cotangent)', fontsize=11)
ax.set_title('A. Baseline structural descriptors', fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.set_yscale('log')

ax = axes[1]
neck = [r for r in baselines if r['mesh_type'] == 'necked_tube']
depths = [float(r['param_value']) for r in neck]
gl2 = [float(r['n_lambda2_graph']) for r in neck]
mu2 = [float(r['n_mu2_cotan']) for r in neck]
ax2 = ax.twinx()
ax.plot(depths, gl2, 'o-', color='#1f77b4', markersize=8, linewidth=2, label=r'$n\lambda_2^G$')
ax2.plot(depths, mu2, 's-', color='#d62728', markersize=8, linewidth=2, label=r'$n\mu_2^C$')
ax.set_xlabel('Neck depth parameter', fontsize=11)
ax.set_ylabel(r'$n \cdot \lambda_2^G$', color='#1f77b4', fontsize=11)
ax2.set_ylabel(r'$n \cdot \mu_2^C$', color='#d62728', fontsize=11)
ax.set_title('B. Necked tube: graph stable,\ncotangent sensitive', fontweight='bold')
ax.tick_params(axis='y', labelcolor='#1f77b4')
ax2.tick_params(axis='y', labelcolor='#d62728')
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=9, loc='center right')

plt.tight_layout()
plt.savefig(f'{OUT}/aisc_fig1_baselines.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 1")


# ═══════════════════════════════════════════════════════════
# Figure 2: Clipping response
# ═══════════════════════════════════════════════════════════
clipping = _load(f'{DATA}/radial_clipping_curves.csv')
selections = [
    ('smooth_tube', '60', 'Smooth tube (n_z=60)'),
    ('necked_tube', '0.8', 'Necked tube (depth=0.8)'),
    ('double_neck_tube', '80', 'Double-neck (n_z=80)'),
    ('branching_tube', '32', 'Branching (n_θ=32)'),
    ('crista_sheet', '16', 'Crista sheet (n_θ=16)'),
    ('crista_sheet', '32', 'Crista sheet (n_θ=32)'),
]
DESC_STYLE = [
    ('retention_lambda2_graph', r'$n\lambda_2^G$', '#1f77b4', '-'),
    ('retention_mu2_cotan', r'$n\mu_2^C$', '#d62728', '-'),
    ('retention_H_degree', r'$H_{deg}$', '#2ca02c', '--'),
    ('retention_H_curvature', r'$H_{curv}$', '#9467bd', '--'),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Radial Clipping Operator Response by Morphology',
             fontweight='bold', fontsize=14, y=1.01)
for idx, (mt, pv, title) in enumerate(selections):
    ax = axes[idx // 3][idx % 3]
    rows = sorted([r for r in clipping if r['mesh_type'] == mt and r['param_value'] == pv],
                   key=lambda r: int(r['clipping_pct']))
    x = [int(r['clipping_pct']) for r in rows]
    for desc, label, color, ls in DESC_STYLE:
        vals = [float(r[desc]) if r[desc] not in (None,'','None') else None for r in rows]
        vx = [xi for xi, v in zip(x, vals) if v is not None]
        vv = [v for v in vals if v is not None]
        ax.plot(vx, vv, f'o{ls}', color=color, markersize=4, linewidth=1.5, label=label)
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax.set_xlabel('Clipping %', fontsize=9)
    ax.set_ylabel('Retention', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    if idx == 0:
        ax.legend(fontsize=7, loc='upper left')
plt.tight_layout()
plt.savefig(f'{OUT}/aisc_fig2_clipping.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 2")


# ═══════════════════════════════════════════════════════════
# Figure 3: Disagreement under clipping
# ═══════════════════════════════════════════════════════════
delta_by_type = defaultdict(lambda: defaultdict(list))
for r in clipping:
    d = r.get('delta_lambda2_mu2', '')
    if d and d not in ('None',''):
        delta_by_type[r['mesh_type']][int(r['clipping_pct'])].append(float(d))

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
for i, mt in enumerate(TYPES):
    if mt not in delta_by_type: continue
    clips = sorted(delta_by_type[mt].keys())
    means = [np.mean(delta_by_type[mt][c]) for c in clips]
    ax.plot(clips, means, 'o-', color=COLORS[i], markersize=6, linewidth=2,
            label=mt.replace('_', ' '))
ax.set_xlabel('Clipping %', fontsize=12)
ax.set_ylabel(r'Mean $\Delta_{\lambda_2, \mu_2}$', fontsize=12)
ax.set_title('Graph–Cotangent Disagreement Under Radial Clipping',
             fontweight='bold', fontsize=13)
ax.legend(fontsize=9)
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{OUT}/aisc_fig3_disagreement.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 3")


# ═══════════════════════════════════════════════════════════
# Figure 4: Real-membrane clipping (from reconstructed CSV)
# ═══════════════════════════════════════════════════════════
real_clip = _load(f'{DATA}/real_membrane_radial_clipping.csv')
mem_types = ['IMM', 'OMM', 'ER']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Real-Membrane Radial Clipping Response (Proof-of-Concept)',
             fontweight='bold', fontsize=14, y=1.02)
for idx, mtype in enumerate(mem_types):
    ax = axes[idx]
    rows = [r for r in real_clip if r['membrane_type'] == mtype]
    by_clip = defaultdict(lambda: {'l2':[], 'm2':[], 'hd':[], 'hc':[]})
    for r in rows:
        cp = int(r['clipping_pct'])
        for key, col in [('l2','retention_lambda2_graph'), ('m2','retention_mu2_cotan'),
                         ('hd','retention_H_degree'), ('hc','retention_H_curvature')]:
            if r[col] and r[col] not in ('None',''):
                by_clip[cp][key].append(float(r[col]))
    clips = sorted(by_clip.keys())
    for key, label, color, ls in [
        ('l2', r'$n\lambda_2^G$', '#1f77b4', '-'),
        ('m2', r'$n\mu_2^C$', '#d62728', '-'),
        ('hd', r'$H_{deg}$', '#2ca02c', '--'),
        ('hc', r'$H_{curv}$', '#9467bd', '--'),
    ]:
        vals = [np.mean(by_clip[c][key]) if by_clip[c][key] else None for c in clips]
        vc = [c for c, v in zip(clips, vals) if v is not None]
        vv = [v for v in vals if v is not None]
        ax.plot(vc, vv, f'o{ls}', color=color, markersize=5, linewidth=1.5, label=label)
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Clipping %', fontsize=10)
    ax.set_ylabel('Retention', fontsize=10)
    mem_colors = {'IMM':'#d62728','OMM':'#1f77b4','ER':'#2ca02c'}
    ax.set_title(mtype, fontsize=12, fontweight='bold', color=mem_colors[mtype])
    if idx == 0:
        ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/aisc_fig4_real_clipping.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 4")


# ═══════════════════════════════════════════════════════════
# Figure 5: NEW — edge-collapse vs random-subsample decimation
# ═══════════════════════════════════════════════════════════
dec_rand = _load(f'{DATA}/decimation_curves_v2.csv')
dec_ec = _load(f'{DATA}/decimation_edge_collapse.csv')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: LCC fraction vs retention, comparing simplifiers
ax = axes[0]
# Random: median LCC per retention level across all 15 meshes × 3 reps
by_ret_rand = defaultdict(list)
for r in dec_rand:
    if r.get('lcc_fraction_of_dec') and r['retention_pct'] != '100':
        by_ret_rand[int(r['retention_pct'])].append(float(r['lcc_fraction_of_dec']))
rets_rand = sorted(by_ret_rand.keys(), reverse=True)
lccs_rand_med = [np.median(by_ret_rand[r]) for r in rets_rand]
lccs_rand_min = [np.min(by_ret_rand[r]) for r in rets_rand]
lccs_rand_max = [np.max(by_ret_rand[r]) for r in rets_rand]

by_ret_ec = defaultdict(list)
for r in dec_ec:
    if r.get('lcc_fraction_of_dec') and r['retention_pct'] != '100':
        by_ret_ec[int(r['retention_pct'])].append(float(r['lcc_fraction_of_dec']))
rets_ec = sorted(by_ret_ec.keys(), reverse=True)
lccs_ec = [np.median(by_ret_ec[r]) for r in rets_ec]

ax.fill_between(rets_rand, lccs_rand_min, lccs_rand_max,
                alpha=0.2, color='#d62728', label='Random subsample (range)')
ax.plot(rets_rand, lccs_rand_med, 'o-', color='#d62728', markersize=8,
        linewidth=2, label='Random subsample (median)')
ax.plot(rets_ec, lccs_ec, 's-', color='#1f77b4', markersize=8,
        linewidth=2, label='Edge-collapse')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Retention %', fontsize=11)
ax.set_ylabel('Largest connected component fraction', fontsize=11)
ax.set_title('A. Topology preservation by simplifier', fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(100, 10)
ax.set_ylim(-0.05, 1.1)

# Panel B: Δ(λ₂,μ₂) vs retention, both simplifiers, averaged across morphologies
ax = axes[1]
delta_rand = defaultdict(list)
for r in dec_rand:
    d = r.get('delta_lambda2_mu2', '')
    if d and d not in ('None','','0.0'):
        delta_rand[int(r['retention_pct'])].append(float(d))
delta_ec = defaultdict(list)
for r in dec_ec:
    d = r.get('delta_lambda2_mu2', '')
    if d and d not in ('None','','0.0'):
        delta_ec[int(r['retention_pct'])].append(float(d))

rets_rand_d = sorted(delta_rand.keys(), reverse=True)
rets_ec_d = sorted(delta_ec.keys(), reverse=True)
drand_med = [np.median(delta_rand[r]) for r in rets_rand_d]
dec_med = [np.median(delta_ec[r]) for r in rets_ec_d]

ax.plot(rets_rand_d, drand_med, 'o-', color='#d62728', markersize=8,
        linewidth=2, label='Random subsample')
ax.plot(rets_ec_d, dec_med, 's-', color='#1f77b4', markersize=8,
        linewidth=2, label='Edge-collapse')
ax.set_xlabel('Retention %', fontsize=11)
ax.set_ylabel(r'Median $\Delta_{\lambda_2, \mu_2}$', fontsize=11)
ax.set_title('B. Graph–cotangent disagreement under simplification',
             fontweight='bold')
ax.legend(fontsize=9)
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlim(100, 10)

plt.tight_layout()
plt.savefig(f'{OUT}/aisc_fig5_simplifier_comparison.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 5")

print("\nAll figures in", OUT)
