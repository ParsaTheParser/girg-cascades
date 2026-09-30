import numpy as np
import csv
import time
from multiprocessing import Pool
from girg_core import (build_GIRG, sample_weights, sample_labels, R0_limit,
                       R0_finite, run_cascade, max_radius)

# Experiment 2: the subcritical results, Theorem 3.3 and Corollary 3.4,
# when R0 < 1. Both are statements conditional on the weight sequence w,
# with the expectation taken over positions, labels, coins and the seed. So
# for each setting we sample ONE weight sequence, keep it fixed, and
# resample the positions and labels for every graph.
#
# Bounds. Theorem 3.3 (proved by counting transmission paths) gives
#     E[Z_s | w] <= (W_n/n) (R0^(n))^s   for every s >= 0,
# for ANY fixed weight sequence with r_uv < 1/2 for all pairs, with no
# (1 + o(1)) factor and no stopping time. Corollary 3.4 then gives
#     P(I_s nonempty | w)  <= (W_n/n)/w_min * (R0^(n))^s,
#     P(|C| > a | w)       <= (W_n/n)/(w_min (1 - R0^(n)) a),
#     P(T_delta < inf | w) <= (W_n/n)/((1 - R0^(n)) n^delta).
# So every dashed curve below is a rigorous upper bound at finite n, computed
# with this w's own R0^(n), W_n/n and w_min. The plotted values are Monte
# Carlo estimates, so they can sit slightly above a bound by sampling error.
# Since Z_s 1{s - 1 < T_delta} <= Z_s, the first bound also covers the
# stopped quantity.
#
# Error bars. Cascades on the same graph are not independent, so every
# standard error below is computed from the spread of the per-graph means
# across the independently generated graphs, not from individual cascades.

np.random.seed(42) # Fixed seed so that every run is reproducible.

m, b_plus, b_minus = 3, 1.2, 0.9 # Same labels as Experiment 1, B = 1.
beta = 4.0
delta = 0.5 # Level n^delta for the hitting probability; any delta > 0 is allowed.
MAX_GEN = 60 # Generations stored per run, far more than any run lasts.

N_GRAPHS = 50
N_TRIALS = 2000 # Per graph -> 100000 cascades per setting.

# Part A: decay and extinction for several R0 < 1 and d = 1, 2, 3.
n_A = 10**5
settings_A = [(d, R0) for d in [1, 2, 3] for R0 in [0.5, 0.7, 0.9]]

# Part B: dependence on n at fixed d = 2 and R0 = 0.9, for the probability
# of reaching n^delta and for the distributions of size and extinction time.
N_GRAPHS_B = 100 # -> 200000 cascades per n.
d_B, R0_B = 2, 0.9
n_B = [10**3, 3 * 10**3, 10**4, 3 * 10**4, 10**5, 3 * 10**5]


def graph_se(per_graph):
    # Standard error of the overall mean, from the spread of the per-graph
    # means. Every graph has the same number of cascades, so the overall
    # mean is the average of the per-graph means.
    per_graph = np.asarray(per_graph)
    return per_graph.std(axis=0, ddof=1) / np.sqrt(per_graph.shape[0])


def run_setting(job):
    d, R0, n, n_graphs, job_id = job
    np.random.seed(42 + job_id)
    tau = R0 / R0_limit(d, beta, 1.0, m, b_plus, b_minus)
    w = sample_weights(n, beta) # One fixed weight sequence per setting.
    # Theorem 3.3 needs r_uv < 1/2 for all pairs, and the Corollary 3.4
    # bounds need R0^(n) < 1. Both hold with high probability over w. With
    # beta = 4 the sum of w_v^2 is heavy tailed, so a rare sample can push
    # R0^(n) above 1. We resample in that case, which is the same as
    # conditioning on the good event of the theorem. This selection rule is
    # stated in the paper's methods.
    while (R0_finite(w, w.sum(), d, tau, m, b_plus, b_minus) >= 1
           or max_radius(w, d) >= 0.5):
        w = sample_weights(n, beta)
    W_n = w.sum()
    level = n**delta

    # Z_graph[g, s] adds up Z_s 1{s - 1 < T_delta} over the runs on graph g
    # (the stopped quantity plotted in the paper; for s = 0 it is just Z_0),
    # and Zall_graph[g, s] adds up Z_s itself, the quantity bounded in
    # Theorem 3.3. Keeping both per graph gives standard errors for both.
    Z_graph = np.zeros((n_graphs, MAX_GEN + 1))
    Zall_graph = np.zeros((n_graphs, MAX_GEN + 1))
    Z_cnt = np.zeros(MAX_GEN + 1) # Number of runs with Z_s > 0.
    sizes, ext_times, hit = [], [], []
    for g in range(n_graphs):
        indptr, indices, _, _ = build_GIRG(n, d, beta, weights=w)
        labels = sample_labels(n, m)
        for _ in range(N_TRIALS):
            c, z = run_cascade(indptr, indices, w, labels, tau, b_plus, b_minus)
            z = z[:MAX_GEN + 1]
            cum = np.cumsum(z)
            # T_delta is the first generation at which the cumulative
            # infected weight reaches n^delta, or infinity if it never does.
            above = np.where(cum >= level)[0]
            T = above[0] if above.size else np.inf
            s = np.arange(len(z))
            # Z_s counts towards the theorem's sum only when s - 1 < T_delta.
            zi = z * (s - 1 < T)
            Z_graph[g, :len(z)] += zi
            Z_cnt[:len(z)] += zi > 0
            Zall_graph[g, :len(z)] += z
            sizes.append(c.sum())
            # Extinction time = first generation with no new infections,
            # which is the number of generations that had infections.
            ext_times.append(len(c))
            hit.append(np.isfinite(T))
    runs = n_graphs * N_TRIALS
    Z_graph /= N_TRIALS # Per-graph means.
    Zall_graph /= N_TRIALS
    ext = np.array(ext_times)
    # Runs are stored graph by graph, so reshaping gives one row per graph.
    alive_graph = np.stack([(ext > s).reshape(n_graphs, N_TRIALS).mean(axis=1)
                            for s in range(MAX_GEN + 1)], axis=1)
    hit_graph = np.array(hit).reshape(n_graphs, N_TRIALS).mean(axis=1)
    # Part B jobs have job_id >= 100, which we use to tell the two apart.
    part = 'B' if job_id >= 100 else 'A'
    return dict(part=part, d=d, R0=R0, n=n, tau=tau, runs=runs, graphs=n_graphs,
                R0_n=R0_finite(w, W_n, d, tau, m, b_plus, b_minus),
                Wn_over_n=W_n / n, w_min=w.min(), w_max=w.max(),
                r_max=max_radius(w, d),
                Z_ind=Z_graph.mean(axis=0), Z_se=graph_se(Z_graph),
                Z_all=Zall_graph.mean(axis=0), Z_all_se=graph_se(Zall_graph),
                Z_cnt=Z_cnt,
                P_alive=alive_graph.mean(axis=0), P_alive_se=graph_se(alive_graph),
                P_T=hit_graph.mean(), P_T_se=graph_se(hit_graph),
                sizes=np.array(sizes), ext=ext)


if __name__ == '__main__':
    t0 = time.time()
    jobs = [(d, R0, n_A, N_GRAPHS, i) for i, (d, R0) in enumerate(settings_A)]
    jobs += [(d_B, R0_B, n, N_GRAPHS_B, 100 + i) for i, n in enumerate(n_B)]
    with Pool(2) as pool:
        res = []
        for r in pool.imap_unordered(run_setting, jobs):
            res.append(r)
            print(f"  d={r['d']} R0={r['R0']} n={r['n']}: R0_n={r['R0_n']:.3f} "
                  f"max r_uv={r['r_max']:.3f} "
                  f"P(T<inf)={r['P_T']:.2e} mean|C|={r['sizes'].mean():.2f} "
                  f"max|C|={r['sizes'].max()} ({time.time() - t0:.0f}s)", flush=True)

    # One row per setting and generation, with the empirical values and the
    # upper bounds of Theorem 3.3 and Corollary 3.4 at lambda = R0^(n).
    with open('decay_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['part', 'd', 'R0', 'n', 'R0_n', 'Wn_over_n', 'w_min', 'runs', 'graphs',
                         'generation', 'E_Zs_ind', 'E_Zs_ind_se', 'runs_with_Zs', 'E_Zs',
                         'E_Zs_se', 'ref_Zs', 'P_alive', 'P_alive_se', 'ref_alive'])
        for r in sorted(res, key=lambda r: (r['part'], r['n'], r['d'], r['R0'])):
            lam = r['R0_n']
            for s in range(MAX_GEN + 1):
                if r['P_alive'][s] == 0 and r['Z_ind'][s] == 0:
                    break
                writer.writerow([r['part'], r['d'], r['R0'], r['n'], lam, r['Wn_over_n'], r['w_min'],
                                 r['runs'], r['graphs'], s,
                                 r['Z_ind'][s], r['Z_se'][s], r['Z_cnt'][s], r['Z_all'][s],
                                 r['Z_all_se'][s],
                                 lam**s * r['Wn_over_n'], r['P_alive'][s], r['P_alive_se'][s],
                                 r['Wn_over_n'] / r['w_min'] * lam**s])

    # Tail of the cascade size, P(|C| > a), with the Markov bound
    # E[|C| | w]/a from Corollary 3.4 at lambda = R0^(n).
    a_grid = np.unique(np.round(np.logspace(0, 3, 40)).astype(int))
    with open('size_tail_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['part', 'd', 'R0', 'n', 'a', 'P_size_gt_a', 'ref'])
        for r in sorted(res, key=lambda r: (r['part'], r['n'], r['d'], r['R0'])):
            lam = r['R0_n']
            for a in a_grid:
                writer.writerow([r['part'], r['d'], r['R0'], r['n'], a, np.mean(r['sizes'] > a),
                                 r['Wn_over_n'] / ((1 - lam) * a * r['w_min'])])

    # Summary per setting: P(T_delta < infinity) with its upper bound,
    # quantiles of the cascade size and extinction time, and the probability
    # of still being alive at generation ceil(log n) with the upper bound
    # from Corollary 3.4 at C = 1.
    with open('summary_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['part', 'd', 'R0', 'n', 'R0_n', 'runs', 'graphs', 'P_T_finite', 'P_T_finite_se',
                         'ref_T', 'mean_size', 'q99_size', 'q999_size', 'max_size',
                         'mean_ext', 'q99_ext', 'q999_ext', 'max_ext',
                         's_logn', 'P_alive_logn', 'ref_alive_logn', 'max_r_uv'])
        for r in sorted(res, key=lambda r: (r['part'], r['n'], r['d'], r['R0'])):
            lam = r['R0_n']
            first = r['Wn_over_n'] / ((1 - lam) * r['n']**delta)
            s_log = int(np.ceil(np.log(r['n'])))
            writer.writerow([r['part'], r['d'], r['R0'], r['n'], lam, r['runs'], r['graphs'],
                             r['P_T'], r['P_T_se'], first,
                             r['sizes'].mean(), np.quantile(r['sizes'], 0.99),
                             np.quantile(r['sizes'], 0.999), r['sizes'].max(),
                             r['ext'].mean(), np.quantile(r['ext'], 0.99),
                             np.quantile(r['ext'], 0.999), r['ext'].max(),
                             s_log, np.mean(r['ext'] > s_log),
                             r['Wn_over_n'] / r['w_min'] * lam**s_log, r['r_max']])

    # Raw sizes and extinction times for the n-scaling runs, for the
    # distribution plots.
    with open('distribution_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n', 'size', 'extinction_time', 'count'])
        for r in sorted(res, key=lambda r: r['n']):
            if r['part'] != 'B':
                continue
            pairs, cnt = np.unique(np.stack([r['sizes'], r['ext']]), axis=1, return_counts=True)
            for (sz, et), c in zip(pairs.T, cnt):
                writer.writerow([r['n'], sz, et, c])
    print(f"All done in {time.time() - t0:.0f}s", flush=True)