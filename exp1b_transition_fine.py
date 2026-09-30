import numpy as np
import csv
import time
from multiprocessing import Pool
from exp1_transition import sweep_point, m, b_plus, b_minus, n_vals
from girg_core import R0_limit

# Finer sweep around the transitions found by exp1_transition.py. The
# coarse grid there has steps of about 0.4 in R0, which is too wide to pin
# down where large cascades start. Here we use steps of 0.1 in the window
# where the coarse sweep showed the change. Same model, same parameters,
# same function; only the R0 grid is different.

windows = [(2, 4.0, 2.4, 4.0, n_vals),
           (3, 4.0, 1.0, 2.0, n_vals),
           (2, 3.5, 1.8, 3.8, [10**4, 10**5]),
           (2, 5.0, 2.4, 4.0, [10**4, 10**5])]

if __name__ == '__main__':
    t0 = time.time()
    jobs = []
    for d, beta, lo, hi, ns in windows:
        for R0 in np.arange(lo, hi + 1e-9, 0.1):
            tau = R0 / R0_limit(d, beta, 1.0, m, b_plus, b_minus)
            for n in ns:
                # Job ids start at 10000 so no seed is shared with the coarse sweep.
                jobs.append((d, beta, n, tau, 10000 + len(jobs)))
    jobs.sort(key=lambda j: -j[2] * j[0])
    with Pool(2) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(sweep_point, jobs)):
            results.append(r)
            if i % 20 == 0:
                print(f"  {i + 1}/{len(jobs)} points done, {time.time() - t0:.0f}s", flush=True)
    results.sort()
    with open('transition_fine_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['d', 'beta', 'n', 'tau', 'R0', 'R0_n', 'P_large', 'P_large_se', 'mean_small_size'])
        for d, beta, n, tau, r0n, p, se, ms in results:
            writer.writerow([d, beta, n, tau, R0_limit(d, beta, tau, m, b_plus, b_minus), r0n, p, se, ms])
    print(f"All done in {time.time() - t0:.0f}s", flush=True)