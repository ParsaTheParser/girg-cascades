# Information cascades on geometric inhomogeneous random graphs

Code and data for the paper "Information cascades on geometric
inhomogeneous random graphs".

Requirements: Python 3, numpy, scipy, matplotlib.

## Files
- girg_core.py: shared model code (graph construction and cascade);
  imported by the other scripts.
- exp1_transition.py: coarse sweep over R0 (onset of large cascades).
- exp1b_transition_fine.py: finer sweep near the onset.
- exp2_subcritical.py: subcritical runs with fixed weights.
- make_paper_figures.py: figures and Table 1 used in the paper.
- make_figures.py: extended set of figures, including ones not in the paper.
- *.csv: simulation output read by the figure scripts.

## Reproduce
Run the simulations (writes the .csv files; each run is seeded):
  python exp1_transition.py
  python exp1b_transition_fine.py
  python exp2_subcritical.py

Make the figures from the .csv files (no simulation needed):
  python make_paper_figures.py
  python make_figures.py

## License
MIT. If you use this code, please cite the paper.
