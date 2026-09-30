import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# Builds every figure and table from the CSV files written by
# exp1_transition.py and exp2_subcritical.py. Nothing is simulated here,
# so the figures can be restyled without rerunning the experiments.

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 12,
                     'legend.fontsize': 9, 'figure.dpi': 110})

# Colours. Values of n are ordered, so they get one hue from light to dark.
# Unordered groups (R0 values, beta values) get distinct hues in a fixed
# order, and every series also gets its own marker so colour is never the
# only way to tell them apart. Reference curves are dashed lines.
N_COLOURS = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
GROUP_COLOURS = ['#2a78d6', '#eb6834', '#1baf7a']
MARKERS = ['o', 's', '^', 'D']
BOUND = dict(color='#0b0b0b', linestyle='--', linewidth=1.6)
REF = dict(color='#8a8984', linestyle=':', linewidth=1.4)
EPS = 0.01
delta = 0.5
MIN_RUNS = 30 # Fewest runs a plotted Monte Carlo mean may rest on.


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def n_label(n):
    n = int(n)
    e = int(np.floor(np.log10(n)))
    c = n // 10**e
    return rf'$n={c}\cdot10^{e}$' if c != 1 else rf'$n=10^{e}$'


def save(fig, name):
    fig.tight_layout()
    fig.savefig(name + '.png', dpi=200, bbox_inches='tight')
    fig.savefig(name + '.pdf', bbox_inches='tight')
    plt.close(fig)


def crossing(x, y, level):
    # First R0 at which y reaches level, by linear interpolation between
    # the two sampled points around it. Returns nan if y never gets there.
    x, y = np.asarray(x), np.asarray(y)
    above = np.where(y >= level)[0]
    if above.size == 0:
        return np.nan
    i = above[0]
    if i == 0:
        return x[0]
    return x[i - 1] + (level - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1])


# Experiment 1
# The coarse and the fine sweep use the same model and the same function,
# so we merge them into one curve per (d, beta, n).
T = read_csv('transition_data.csv') + read_csv('transition_fine_data.csv')
curves = defaultdict(list)
for r in T:
    curves[(int(r['d']), r['beta'], int(r['n']))].append(r)
for k in curves:
    curves[k].sort(key=lambda r: r['R0'])

# Figure 1: fraction of large cascades against R0, one panel per dimension.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for ax, d in zip(axes, [1, 2, 3]):
    ns = sorted(n for (dd, b, n) in curves if dd == d and b == 4.0)
    for i, n in enumerate(ns):
        c = curves[(d, 4.0, n)]
        x = [r['R0'] for r in c]
        ax.errorbar(x, [r['P_large'] for r in c], yerr=[1.96 * r['P_large_se'] for r in c],
                    fmt=MARKERS[i] + '-', color=N_COLOURS[i], markersize=5, linewidth=1.8,
                    capsize=2, label=n_label(n))
    ax.axvline(1.0, **REF, label=r'$\mathcal{R}_0=1$')
    ax.set_title(rf'$d={d}$, $\beta=4$')
    ax.set_xlabel(r'Mean-field reproduction factor $\mathcal{R}_0$')
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel(rf'P(cascade infects $\geq {EPS:g}\,n$ vertices)')
axes[0].legend(loc='upper left')
save(fig, 'fig1_transition_by_dimension')

# Figure 2: the same for d = 2 and three weight exponents, at n = 10^4 and
# n = 10^5, to see whether the location of the transition depends on beta.
fig, ax = plt.subplots(figsize=(7, 4.6))
for i, beta in enumerate([3.5, 4.0, 5.0]):
    for n, style, fill in [(10**4, '--', 'none'), (10**5, '-', None)]:
        c = curves.get((2, beta, n))
        if not c:
            continue
        ax.errorbar([r['R0'] for r in c], [r['P_large'] for r in c],
                    yerr=[1.96 * r['P_large_se'] for r in c], fmt=MARKERS[i] + style,
                    color=GROUP_COLOURS[i], markerfacecolor=fill, markersize=5,
                    linewidth=1.6, capsize=2, label=rf'$\beta={beta:g}$, ' + n_label(n))
ax.axvline(1.0, **REF, label=r'$\mathcal{R}_0=1$')
ax.set_xlabel(r'Mean-field reproduction factor $\mathcal{R}_0$')
ax.set_ylabel(rf'P(cascade infects $\geq {EPS:g}\,n$ vertices)')
ax.set_title(r'Effect of the weight exponent ($d=2$)')
ax.legend(loc='upper left', ncol=2)
ax.grid(True, alpha=0.3)
save(fig, 'fig2_transition_by_beta')

# Figure 3: mean size of the cascades that stayed small. Below the
# transition this is the whole cascade, so it shows how cascade size
# grows with R0 before large cascades appear.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for ax, d in zip(axes, [1, 2, 3]):
    ns = sorted(n for (dd, b, n) in curves if dd == d and b == 4.0)
    for i, n in enumerate(ns):
        c = curves[(d, 4.0, n)]
        ax.semilogy([r['R0'] for r in c], [r['mean_small_size'] for r in c], MARKERS[i] + '-',
                    color=N_COLOURS[i], markersize=5, linewidth=1.8, label=n_label(n))
    ax.axvline(1.0, **REF, label=r'$\mathcal{R}_0=1$')
    ax.set_title(rf'$d={d}$, $\beta=4$')
    ax.set_xlabel(r'Mean-field reproduction factor $\mathcal{R}_0$')
    ax.grid(True, alpha=0.3, which='both')
axes[0].set_ylabel(rf'Mean $|\mathcal{{C}}|$ of cascades below ${EPS:g}\,n$')
axes[0].legend(loc='upper left')
save(fig, 'fig3_small_cascade_size')

# Table 1: estimated onset of large cascades. For each setting we report
# the R0 at which the fraction of large cascades first reaches 5% and 50%,
# and the R0 at which the mean size of the small cascades peaks. That peak
# is the usual finite-size marker of a critical point: just below it small
# cascades keep growing, just above it the big ones escape and are no
# longer counted as small.
with open('table1_onset.csv', 'w', newline='') as f, open('table1_onset.tex', 'w') as tex:
    writer = csv.writer(f)
    writer.writerow(['d', 'beta', 'n', 'R0_max_tested', 'R0_at_5pct', 'R0_at_50pct', 'R0_at_peak_small_size'])
    tex.write('\\begin{tabular}{ccrcccc}\n\\toprule\n'
              '$d$ & $\\beta$ & $n$ & largest $\\mathcal{R}_0$ tested & '
              '$\\mathcal{R}_0$ at 5\\% & $\\mathcal{R}_0$ at 50\\% & peak of small sizes \\\\\n\\midrule\n')
    for (d, beta, n) in sorted(curves):
        c = curves[(d, beta, n)]
        x = [r['R0'] for r in c]
        y = [r['P_large'] for r in c]
        a, b = crossing(x, y, 0.05), crossing(x, y, 0.5)
        ms = np.array([r['mean_small_size'] for r in c])
        # With no large cascades at all (d = 1) there is no peak inside the
        # tested range, so we leave that entry empty.
        peak = x[int(np.nanargmax(ms))] if np.nanmax(y) >= 0.05 else np.nan
        writer.writerow([d, beta, n, max(x), a, b, peak])
        fmt = lambda v: '--' if np.isnan(v) else f'{v:.2f}'
        tex.write(f'{d} & {beta:g} & {n:,} & {max(x):.2f} & {fmt(a)} & {fmt(b)} & {fmt(peak)} \\\\\n'.replace(',', '{,}'))
    tex.write('\\bottomrule\n\\end{tabular}\n')

# Figure 4: reproduction inside real cascades, E|I_(s+1)| / E|I_s|, against
# the generation s, next to R0 (dashed) and the fresh-environment value
# for an edge-reached vertex (hollow marker at s = 0).
G = read_csv('generation_data.csv')
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for ax, d in zip(axes, [1, 2, 3]):
    R0s = sorted({r['R0'] for r in G if r['d'] == d})
    for i, R0 in enumerate(R0s):
        rows = [r for r in G if r['d'] == d and r['R0'] == R0]
        # Generation 0 is the uniformly chosen seed, which is not degree
        # biased, so its ratio is not comparable with R0 and we start at 1.
        # We also drop generations with fewer than 200 infections in total,
        # where the ratio is too noisy to read.
        pts = [(r['generation'], r['ratio_next']) for r in rows
               if r['generation'] >= 1 and r['sum_I_s'] >= 200 and not np.isnan(r['ratio_next'])]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], MARKERS[i] + '-', color=GROUP_COLOURS[i],
                markersize=6, linewidth=1.8, label=rf'In cascade, $\mathcal{{R}}_0={R0:g}$')
        ax.axhline(R0, color=GROUP_COLOURS[i], linestyle='--', linewidth=1.2)
        ax.errorbar([0], [rows[0]['fresh']], yerr=[1.96 * rows[0]['fresh_se']], fmt=MARKERS[i],
                    color=GROUP_COLOURS[i], markerfacecolor='none', markersize=8, capsize=3,
                    label='Fresh environment' if i == 0 else None)
    ax.axhline(1.0, **REF)
    ax.set_xticks(range(0, 11))
    ax.set_xticklabels(['fresh'] + [str(s) for s in range(1, 11)])
    ax.set_title(rf'$d={d}$, $\beta=4$, $n=10^5$')
    ax.set_xlabel(r'Generation $s$')
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel(r'Reproduction $\mathbb{E}|\mathcal{I}_{s+1}|\,/\,\mathbb{E}|\mathcal{I}_s|$')
axes[0].legend(loc='upper right', fontsize=8)
save(fig, 'fig4_reproduction_by_generation')

# Experiment 2
D = read_csv('decay_data.csv')
S = read_csv('size_tail_data.csv')
M = read_csv('summary_data.csv')

# Figure: Theorem 3.3. Empirical E[Z_s 1{s-1 < T_delta} | w] (markers)
# against the upper bound (W_n/n) (R0^(n))^s of Theorem 3.3 (dashed). The
# bound holds at finite n for the fixed w, and it covers the stopped
# quantity because Z_s 1{s-1 < T_delta} <= Z_s. Points can sit slightly
# above it only through Monte Carlo error.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for ax, d in zip(axes, [1, 2, 3]):
    for i, R0 in enumerate([0.5, 0.7, 0.9]):
        rows = [r for r in D if r['part'] == 'A' and r['d'] == d and r['R0'] == R0]
        s = np.array([r['generation'] for r in rows])
        z = np.array([r['E_Zs_ind'] for r in rows])
        se = np.array([r['E_Zs_ind_se'] for r in rows])
        # We only show generations reached by at least MIN_RUNS runs. Further
        # out the mean is carried by a handful of runs and is pure noise.
        keep = np.array([r['runs_with_Zs'] for r in rows]) >= MIN_RUNS
        ax.errorbar(s[keep], z[keep], yerr=1.96 * se[keep], fmt=MARKERS[i], color=GROUP_COLOURS[i],
                    markersize=6, capsize=2,
                    label=rf'Empirical, $\mathcal{{R}}_0^{{(n)}}={rows[0]["R0_n"]:.3f}$')
        ax.set_yscale('log')
        ax.semilogy(s, [r['ref_Zs'] for r in rows], color=GROUP_COLOURS[i], linestyle='--',
                    linewidth=1.6)
    ax.plot([], [], **BOUND, label=r'Upper bound $(\mathcal{R}_0^{(n)})^sW_n/n$ (Theorem 3.3)')
    ax.set_title(rf'$d={d}$, $n=10^5$')
    ax.set_xlabel(r'Generation $s$')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='lower left', fontsize=8)
axes[0].set_ylabel(r'$\mathbb{E}[Z_s\mathbb{1}_{\{s-1<T_\delta\}}\mid\mathbf{w}]$')
save(fig, 'fig5_decay_bound')

# Figure 6: Corollary 3.4, survival probability P(I_s nonempty | w).
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for ax, d in zip(axes, [1, 2, 3]):
    for i, R0 in enumerate([0.5, 0.7, 0.9]):
        rows = [r for r in D if r['part'] == 'A' and r['d'] == d and r['R0'] == R0]
        s = np.array([r['generation'] for r in rows])
        p = np.array([r['P_alive'] for r in rows])
        se = np.array([r['P_alive_se'] for r in rows])
        keep = p * rows[0]['runs'] >= MIN_RUNS
        ax.errorbar(s[keep], p[keep], yerr=1.96 * se[keep], fmt=MARKERS[i], color=GROUP_COLOURS[i],
                    markersize=6, capsize=2,
                    label=rf'Empirical, $\mathcal{{R}}_0^{{(n)}}={rows[0]["R0_n"]:.3f}$')
        ax.semilogy(s, [r['ref_alive'] for r in rows], color=GROUP_COLOURS[i], linestyle='--',
                    linewidth=1.6)
    ax.plot([], [], **BOUND, label=r'Upper bound (Corollary 3.4), $\lambda=\mathcal{R}_0^{(n)}$')
    ax.set_title(rf'$d={d}$, $n=10^5$')
    ax.set_xlabel(r'Generation $s$')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='lower left', fontsize=8)
axes[0].set_ylabel(r'$\mathbb{P}(\mathcal{I}_s\neq\emptyset\mid\mathbf{w})$')
save(fig, 'fig6_extinction_bound')

# Figure 7: Corollary 3.4, cascade-size tail P(|C| > a | w).
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for ax, d in zip(axes, [1, 2, 3]):
    for i, R0 in enumerate([0.5, 0.7, 0.9]):
        rows = [r for r in S if r['part'] == 'A' and r['d'] == d and r['R0'] == R0]
        a = np.array([r['a'] for r in rows])
        p = np.array([r['P_size_gt_a'] for r in rows])
        runs = [x for x in M if x['part'] == 'A' and x['d'] == d and x['R0'] == R0][0]['runs']
        keep = p * runs >= MIN_RUNS
        ax.loglog(a[keep], p[keep], MARKERS[i], color=GROUP_COLOURS[i], markersize=5,
                  label=rf'Empirical, $\mathcal{{R}}_0={R0:g}$')
        ax.loglog(a, [r['ref'] for r in rows], color=GROUP_COLOURS[i], linestyle='--',
                  linewidth=1.6)
    ax.plot([], [], **BOUND, label=r'Upper bound (Corollary 3.4), $\lambda=\mathcal{R}_0^{(n)}$')
    ax.set_ylim(1e-4, 2)
    ax.set_title(rf'$d={d}$, $n=10^5$')
    ax.set_xlabel(r'Cascade size $a$')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='lower left', fontsize=8)
axes[0].set_ylabel(r'$\mathbb{P}(|\mathcal{C}|>a\mid\mathbf{w})$')
save(fig, 'fig7_size_tail_bound')

# Figure 8: P(T_delta < infinity | w) against n, with the bound
# (W_n/n) / ((1 - lambda) n^delta) from Corollary 3.4.
B = sorted([r for r in M if r['part'] == 'B'], key=lambda r: r['n'])
fig, ax = plt.subplots(figsize=(7, 4.6))
n = np.array([r['n'] for r in B])
p = np.array([r['P_T_finite'] for r in B])
runs = np.array([r['runs'] for r in B])
se = np.array([r['P_T_finite_se'] for r in B]) # Spread across graphs.
keep = p > 0
ax.errorbar(n[keep], p[keep], yerr=1.96 * se[keep], fmt='o', color=GROUP_COLOURS[0],
            markersize=7, capsize=3, label=r'Empirical $\mathbb{P}(T_\delta<\infty\mid\mathbf{w})$')
if (~keep).any():
    # A setting with no hits only tells us the probability is below about
    # 3/runs, so we show that upper limit with a downward triangle.
    ax.plot(n[~keep], 3 / runs[~keep], 'v', color=GROUP_COLOURS[0], markersize=7,
            markerfacecolor='none', label='No hits (95% upper limit)')
ax.plot(n, [r['ref_T'] for r in B], marker='o', **BOUND, markersize=4,
        label=r'Upper bound $\frac{W_n/n}{(1-\lambda)n^{\delta}}$, $\lambda=\mathcal{R}_0^{(n)}$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$n$')
ax.set_ylabel(r'$\mathbb{P}(\mathcal{Z}\ \mathrm{reaches}\ n^{\delta}\mid\mathbf{w})$')
ax.set_title(rf'Hitting the stopping level ($d=2$, $\mathcal{{R}}_0=0.9$, $\delta={delta}$)')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3, which='both')
save(fig, 'fig8_stopping_level_vs_n')

# Figure 9: O_P(1). Tails of the cascade size and of the extinction time for
# growing n at fixed R0 = 0.9. If both are bounded in probability, the
# curves should lie on top of each other instead of moving right with n.
R = read_csv('distribution_data.csv')
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ns = sorted({r['n'] for r in R})
ramp = ['#b7d3f6', '#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']
for i, nn in enumerate(ns):
    rows = [r for r in R if r['n'] == nn]
    total = sum(r['count'] for r in rows)
    for ax, key, grid in [(axes[0], 'size', np.unique(np.round(np.logspace(0, 3, 50)))),
                          (axes[1], 'extinction_time', np.arange(1, 40))]:
        vals = np.array([r[key] for r in rows])
        cnt = np.array([r['count'] for r in rows])
        tail = np.array([cnt[vals > g].sum() / total for g in grid])
        keep = tail > 0
        ax.plot(grid[keep], tail[keep], '-', color=ramp[i], linewidth=1.8,
                marker=['o', 's', '^', 'D', 'v', 'P'][i], markevery=4, markersize=5,
                label=n_label(nn))
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_xlabel(r'Cascade size $a$')
axes[0].set_ylabel(r'$\mathbb{P}(|\mathcal{C}|>a)$')
axes[0].set_title(r'Cascade size ($d=2$, $\mathcal{R}_0=0.9$)')
axes[1].set_yscale('log')
axes[1].set_xlabel(r'Generation $s$')
axes[1].set_ylabel(r'$\mathbb{P}(\mathrm{extinction\ time}>s)$')
axes[1].set_title(r'Extinction time ($d=2$, $\mathcal{R}_0=0.9$)')
for ax in axes:
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='lower left')
save(fig, 'fig9_OP1_distributions')

# Table 2: summary of the subcritical runs.
with open('table2_subcritical.tex', 'w') as tex:
    tex.write('\\begin{tabular}{ccrcccccc}\n\\toprule\n'
              '$d$ & $\\mathcal R_0$ & $n$ & $\\mathcal R_0^{(n)}$ & '
              '$\\hat{\\mathbb P}(T_\\delta<\\infty)$ & bound & '
              'mean $|\\mathcal C|$ & 99.9\\% $|\\mathcal C|$ & '
              '$\\hat{\\mathbb P}(\\mathcal I_{\\lceil\\log n\\rceil}\\neq\\varnothing)$ \\\\\n\\midrule\n')
    for r in sorted(M, key=lambda r: (r['part'], r['d'], r['R0'], r['n'])):
        tex.write(f"{int(r['d'])} & {r['R0']:g} & {int(r['n']):,} & {r['R0_n']:.3f} & "
                  f"{r['P_T_finite']:.1e} & {r['ref_T']:.1e} & {r['mean_size']:.2f} & "
                  f"{r['q999_size']:.0f} & {r['P_alive_logn']:.1e} \\\\\n".replace(',', '{,}'))
    tex.write('\\bottomrule\n\\end{tabular}\n')
print('Figures and tables written.')