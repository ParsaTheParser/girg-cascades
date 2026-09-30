import numpy as np
import csv
import sys
import time
from multiprocessing import Pool
from girg_core import (build_GIRG, sample_labels, R0_limit, R0_finite,
                       run_cascade, fresh_offspring)

# Experiment 1: where does the transition to large cascades actually happen?
# We sweep tau so that the limiting R0 of Lemma 3.1 runs over a range of
# values, and for each value we record how often a cascade becomes large.
# If R0 = 1 were a sharp threshold, large cascades would start appearing
# right after R0 = 1. The second part records the reproduction inside
# real cascades, generation by generation, to compare with R0.

np.random.seed(42) # Fixed seed so that every run is reproducible.

# Label parameters, kept identical in both experiments. With m = 3 labels,
# b^+ = 1.2 and b^- = 0.9 we get B = 1.2/3 + (2/3)0.9 = 1.0, and the
# condition tau b^+ <= 1 allows tau up to 1/1.2, so R0 can be pushed
# well above 1 in every setting below.
m, b_plus, b_minus = 3, 1.2, 0.9
tau_max = 1 / b_plus
EPS = 0.01 # A cascade counts as large once it infects at least EPS * n vertices.

N_GRAPHS = 8 # Fresh graphs (weights, positions, labels) per point.
N_TRIALS = 250 # Cascades per graph -> 2000 cascades per point.

# Main sweep: beta = 4 in d = 1, 2, 3, plus beta = 3.5 and 5 in d = 2 to
# see whether the location of the transition depends on the weight tail.
settings = [(1, 4.0), (2, 4.0), (3, 4.0), (2, 3.5), (2, 5.0)]
n_vals = [10**4, 3 * 10**4, 10**5, 3 * 10**5]
N_POINTS = 18


def sweep_point(job):
    d, beta, n, tau, job_id = job
    np.random.seed(42 + job_id) # Separate reproducible stream per job.
    large, sizes, r0n = [], [], []
    for _ in range(N_GRAPHS):
        indptr, indices, w, W_n = build_GIRG(n, d, beta)
        labels = sample_labels(n, m)
        r0n.append(R0_finite(w, W_n, d, tau, m, b_plus, b_minus))
        for _ in range(N_TRIALS):
            # We stop a cascade once it is large. Its final size is then
            # unknown, but whether it became large is all we need here.
            c, _ = run_cascade(indptr, indices, w, labels, tau, b_plus, b_minus,
                               max_size=int(EPS * n))
            sizes.append(c.sum())
            large.append(c.sum() >= EPS * n)
    large = np.array(large)
    p = large.mean()
    # Cascades on the same graph are not independent, so the standard error
    # comes from the spread of the per-graph fractions across the N_GRAPHS
    # independent graphs. Runs are stored graph by graph, so reshaping gives
    # one row per graph.
    p_graph = large.reshape(N_GRAPHS, N_TRIALS).mean(axis=1)
    se = p_graph.std(ddof=1) / np.sqrt(N_GRAPHS)
    # Mean size among the cascades that stayed small, which shows how the
    # small cascades grow as R0 increases.
    small = np.array(sizes)[~large]
    mean_small = small.mean() if small.size else np.nan
    return d, beta, n, tau, np.mean(r0n), p, se, mean_small


# Per-generation reproduction. For these runs we do not stop cascades
# early, so that every generation is recorded in full.
N_GRAPHS_G = 20
N_TRIALS_G = 1000
N_FRESH = 5000 # Edge-reached vertices sampled per graph for the fresh value.
n_gen = 10**5
MAX_GEN = 10
R0_gen = [0.8, 1.5, 2.5]


def generation_point(job):
    d, beta, R0, job_id = job
    np.random.seed(4242 + job_id)
    tau = R0 / R0_limit(d, beta, 1.0, m, b_plus, b_minus)
    # sum_counts[s] adds up |I_s| over all runs. Runs that died out before
    # generation s add 0, so the ratio of consecutive sums is
    # E|I_(s+1)| / E|I_s| over all runs, not just the surviving ones.
    sum_counts = np.zeros(MAX_GEN + 2)
    fresh, r0n = [], []
    for _ in range(N_GRAPHS_G):
        indptr, indices, w, W_n = build_GIRG(n_gen, d, beta)
        labels = sample_labels(n_gen, m)
        r0n.append(R0_finite(w, W_n, d, tau, m, b_plus, b_minus))
        fresh.append(fresh_offspring(indptr, indices, labels, tau, b_plus,
                                     b_minus, N_FRESH).mean())
        for _ in range(N_TRIALS_G):
            c, _ = run_cascade(indptr, indices, w, labels, tau, b_plus, b_minus)
            k = min(len(c), MAX_GEN + 2)
            sum_counts[:k] += c[:k]
    return d, beta, R0, np.mean(r0n), np.mean(fresh), np.std(fresh) / np.sqrt(N_GRAPHS_G), sum_counts


if __name__ == '__main__':
    t0 = time.time()
    jobs = []
    for d, beta in settings:
        R0_max = R0_limit(d, beta, tau_max, m, b_plus, b_minus)
        # The sweep starts below 1 and runs up to the largest R0 allowed by
        # tau b^+ <= 1 (capped at 8, far beyond any transition we see).
        for R0 in np.linspace(0.5, min(R0_max, 8.0), N_POINTS):
            tau = R0 / R0_limit(d, beta, 1.0, m, b_plus, b_minus)
            ns = n_vals if beta == 4.0 else [10**4, 10**5]
            for n in ns:
                jobs.append((d, beta, n, tau, len(jobs)))
    # Big jobs first so the two workers finish at about the same time.
    jobs.sort(key=lambda j: -j[2] * j[0])

    with Pool(2) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(sweep_point, jobs)):
            results.append(r)
            if i % 20 == 0:
                print(f"  {i + 1}/{len(jobs)} points done, {time.time() - t0:.0f}s", flush=True)
    results.sort()
    with open('transition_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['d', 'beta', 'n', 'tau', 'R0', 'R0_n', 'P_large', 'P_large_se', 'mean_small_size'])
        for d, beta, n, tau, r0n, p, se, ms in results:
            writer.writerow([d, beta, n, tau, R0_limit(d, beta, tau, m, b_plus, b_minus), r0n, p, se, ms])
    print(f"Sweep done in {time.time() - t0:.0f}s", flush=True)
    # Run with --sweep-only to redo the sweep without the per-generation
    # part, whose output (generation_data.csv) does not depend on the sweep.
    if '--sweep-only' in sys.argv:
        sys.exit()

    gjobs = [(d, 4.0, R0, i) for i, (d, R0) in
             enumerate([(d, R0) for d in [1, 2, 3] for R0 in R0_gen])]
    with Pool(2) as pool:
        gres = pool.map(generation_point, gjobs)
    with open('generation_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['d', 'beta', 'R0', 'R0_n', 'fresh', 'fresh_se', 'generation', 'sum_I_s', 'ratio_next'])
        for d, beta, R0, r0n, fr, fse, sc in gres:
            for s in range(MAX_GEN + 1):
                ratio = sc[s + 1] / sc[s] if sc[s] > 0 else np.nan
                writer.writerow([d, beta, R0, r0n, fr, fse, s, sc[s], ratio])
    print(f"All done in {time.time() - t0:.0f}s", flush=True)