"""
Phase 8 figures + LaTeX table rows for the revised paper.

Outputs:
  fig_cost_crossover.pdf   (a) mean expected cost vs cost ratio incl. AllPositive
                           (b) per-dataset crossover ratio r* vs IR
  fig_cd_diagram_tuned.pdf CD diagram for the equal-budget tuned comparison (F1)
  stdout: LaTeX rows for tab:tuned, tab:weight_schemes, tab:by_ir
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 10, 'font.family': 'serif',
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1
})

COST_RATIOS = [1, 2, 5, 10, 20, 50]
COST_COLS = [f'cost_r{r}' for r in COST_RATIOS]

tuned = pd.read_csv('phase9_tuned_results.csv')          # 13 methods, tuned, 3 seeds
naive = pd.read_csv('phase8_naive_rb_results.csv')       # naive + RB + ASE refs
weights = pd.read_csv('phase8_weighting_results.csv')    # 9 weighting schemes

# ---------------------------------------------------------------------------
# Figure: cost crossover
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Panel (a): mean expected cost vs ratio
sel = {
    'ASE-Stacking': ('tab:blue', '-', 'o'),
    'ASE-Static': ('tab:cyan', '-', 's'),
    'SPE': ('tab:red', '--', 'P'),
    'RandomBalance': ('tab:green', '--', '^'),
    'SMOTEBoost': ('tab:orange', '--', 'v'),
    'EasyEnsemble': ('tab:purple', '--', 'd'),
}
tm = tuned.groupby('Method')[COST_COLS].mean()
for m, (c, ls, mk) in sel.items():
    ax1.plot(COST_RATIOS, tm.loc[m, COST_COLS], ls, color=c, marker=mk,
             ms=4, label=m.replace('ASE', 'SE'))

try:
    rm = pd.read_csv('phase9_ratio_matched.csv')
    mt = rm[rm.Scheme == 'stacking_matched'].groupby('Scheme')[COST_COLS].mean().iloc[0]
    ax1.plot(COST_RATIOS, mt, '-', color='navy', marker='*', ms=7, lw=1.8,
             label='SE-Stacking (ratio-matched)')
except FileNotFoundError:
    pass

ap = naive[naive.Scheme == 'AllPositive'].groupby('Scheme')[COST_COLS].mean().iloc[0]
ax1.plot(COST_RATIOS, ap, '-', color='black', lw=2, label='All-Positive (naive)')

ax1.axvspan(5, 30, alpha=0.10, color='tab:blue')
ax1.text(12.2, 0.135, 'target regime\n$5 \\leq \\rho \\leq 30$', ha='center',
         fontsize=8, color='tab:blue')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xticks(COST_RATIOS)
ax1.set_xticklabels([str(r) for r in COST_RATIOS])
ax1.set_xlabel(r'Cost ratio $\rho = C_{FN}/C_{FP}$')
ax1.set_ylabel('Mean expected cost per instance')
ax1.set_title('(a) Expected cost vs. cost ratio')
ax1.legend(frameon=False, fontsize=8)

# Panel (b): per-dataset crossover ratio r* (SE-Stacking vs AllPositive) vs IR
# cost_m(r) = a + b r  (a = FP/n, b = FN/n); AllPositive cost = const = a_AP.
# r* = (a_AP - a_m)/b_m
def linfit(row):
    y = row[COST_COLS].values.astype(float)
    b = (y[-1] - y[0]) / (COST_RATIOS[-1] - COST_RATIOS[0])
    a = y[0] - b * COST_RATIOS[0]
    return a, b

xs, ys = [], []
stk = tuned[tuned.Method == 'ASE-Stacking'].set_index('Dataset')
apn = naive[naive.Scheme == 'AllPositive'].set_index('Dataset')
for ds in stk.index:
    a_m, b_m = linfit(stk.loc[ds])
    a_ap, _ = linfit(apn.loc[ds])
    ir = stk.loc[ds, 'IR']
    rstar = (a_ap - a_m) / b_m if b_m > 1e-9 else np.inf
    xs.append(ir)
    ys.append(min(rstar, 500))

xs, ys = np.array(xs), np.array(ys)
finite = ys < 500
ax2.scatter(xs[finite], ys[finite], s=28, color='tab:blue', zorder=3,
            label='crossover $\\rho^*$')
ax2.scatter(xs[~finite], np.full((~finite).sum(), 500), s=36, marker='^',
            color='tab:blue', zorder=3, label='$\\rho^* > 500$ (FN $\\approx$ 0)')
ax2.axhspan(5, 30, alpha=0.10, color='tab:blue')
med = np.median(ys[finite])
ax2.axhline(med, color='gray', ls=':', lw=1)
ax2.text(1.15, med * 1.15, f'median $\\rho^*$ = {med:.0f}', fontsize=8, color='gray')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Imbalance ratio (IR)')
ax2.set_ylabel('Crossover cost ratio $\\rho^*$')
ax2.set_title('(b) Where All-Positive overtakes SE-Stacking')
ax2.legend(frameon=False, fontsize=8, loc='lower right')

plt.tight_layout()
plt.savefig('fig_cost_crossover.pdf')
plt.close()
print('fig_cost_crossover.pdf saved')
print(f'  median rho* = {med:.1f}; rho* < 5 en {int((ys[finite] < 5).sum())} datasets;'
      f' rho* entre 5-30 en {int(((ys[finite] >= 5) & (ys[finite] <= 30)).sum())};'
      f' > 30 en {int((ys >= 30).sum())}')

# ---------------------------------------------------------------------------
# Figure: CD diagram (tuned, F1)
# ---------------------------------------------------------------------------
from create_cd_diagram import create_cd_diagram

pivot = tuned.pivot_table(index='Dataset', columns='Method', values='F1')
ranks = pivot.rank(axis=1, ascending=False).mean()
n_methods = len(ranks)
n_datasets = pivot.shape[0]
q_alpha = {10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313}
cd = q_alpha[n_methods] * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
create_cd_diagram({m.replace('ASE', 'SE'): r for m, r in ranks.items()}, cd,
                  output_file='fig_cd_diagram_tuned.pdf')
print(f'fig_cd_diagram_tuned.pdf saved (CD={cd:.3f})')

# ---------------------------------------------------------------------------
# LaTeX rows
# ---------------------------------------------------------------------------
print('\n===== tab:tuned (mean over 22 datasets, equal-budget tuning) =====')
cols = ['Recall', 'Precision', 'F1', 'AUC', 'cost_r5', 'cost_r20']
agg = tuned.groupby('Method')[cols].mean()
rank_recall = tuned.pivot_table(index='Dataset', columns='Method', values='Recall') \
                   .rank(axis=1, ascending=False).mean()
rank_f1 = ranks
rank_c5 = tuned.pivot_table(index='Dataset', columns='Method', values='cost_r5') \
               .rank(axis=1, ascending=True).mean()
agg['rank_Recall'] = rank_recall
agg['rank_F1'] = rank_f1
agg['rank_cost5'] = rank_c5
agg = agg.sort_values('rank_cost5')
for m, r in agg.iterrows():
    name = m.replace('ASE-', 'SE-')
    print(f"{name} & {r.Recall:.3f} & {r.Precision:.3f} & {r.F1:.3f} & "
          f"{r.cost_r5:.3f} & {r.cost_r20:.3f} & {r.rank_Recall:.2f} & "
          f"{r.rank_F1:.2f} & {r.rank_cost5:.2f} \\\\")

print('\n===== tab:weight_schemes (mean over 22 datasets) =====')
worder = ['uniform', 'type', 'type_recall_train', 'type_recall_oof', 'type_f1_oof',
          'type_gmean_oof', 'type_bacc_oof', 'bias_prop_oof', 'stacking_cost']
wagg = weights.groupby('Scheme')[['Recall', 'Precision', 'F1', 'cost_r5', 'cost_r20']].mean()
wr = weights.pivot_table(index='Dataset', columns='Scheme', values='Recall') \
            .rank(axis=1, ascending=False).mean()
wc = weights.pivot_table(index='Dataset', columns='Scheme', values='cost_r5') \
            .rank(axis=1, ascending=True).mean()
for s in worder:
    r = wagg.loc[s]
    print(f"{s} & {r.Recall:.3f} & {r.Precision:.3f} & {r.F1:.3f} & "
          f"{r.cost_r5:.3f} & {r.cost_r20:.3f} & {wr[s]:.2f} & {wc[s]:.2f} \\\\")

print('\n===== tab:by_ir (mean F1 / Recall by IR category, tuned) =====')
tuned['IRcat'] = pd.cut(tuned.IR, bins=[0, 3, 10, 30, 1000],
                        labels=['Low', 'Medium', 'High', 'VeryHigh'])
for metric in ['F1', 'Recall']:
    print(f'--- {metric} ---')
    t = tuned.groupby(['Method', 'IRcat'], observed=True)[metric].mean().unstack()
    t = t.loc[agg.index]
    for m, r in t.iterrows():
        name = m.replace('ASE-', 'SE-')
        print(f"{name} & {r.Low:.3f} & {r.Medium:.3f} & {r.High:.3f} & {r.VeryHigh:.3f} \\\\")
