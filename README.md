# The Learned Light Cone of Scientific Emulators

Physics limits how far information can travel during one forecast step. This repository measures
whether scientific emulators respect that limit by comparing their response to localized
perturbations with a prescribed physical cone.

## Paper

Learned Light Cones: Controlling the Information Reach of Scientific Emulators.
Arnav Garg. arXiv identifier to be added.

## Headline results

- Across 6 released checkpoints, out-of-cone response energy separates into two groups: two
  checkpoints sit at 20.9 and 5.83 percent, while the others are at or below 0.70 percent.
  A separate measured local-convolution reference, paired over the same samples, sits at 0.46 percent
  (`results/tables/deployed/deployed_fleet.csv`; `results/tables/summary.csv`).
- Cone-matched local mixing helps wherever it was tested. Masking out-of-cone inputs raises the
  trained global operator's error by a factor of 415 against 1.00 for the matched local control,
  the local arm of the two-dimensional headline comparison reaches 0.032 out-of-distribution MSE
  against 0.663 while using 27.6 times fewer real parameters, support-restricted arms win all 36
  one-dimensional equal-width draws, and a local ERA5 configuration lowers six-hour Z500 error by
  36.6 percent (`results/tables/summary.csv`).
- The direct 24-hour Pangu graph has a static certified-zero region covering 11.57 to 43.45
  percent of output area across four fixed source columns
  (`results/tables/deployed/compiled_reach_pangu.json`).

## Install and reproduce

Install [uv](https://docs.astral.sh/uv/), clone the repository, and run:

```bash
bash reproduce.sh
```

The script provisions the locked environment, runs the tests, verifies retained records, and
regenerates the derived tables and figures. A clean Git worktree is required.

## Outputs

- [`results/tables/summary.csv`](results/tables/summary.csv): compact headline results
- [`results/tables/robustness_summary.csv`](results/tables/robustness_summary.csv): robustness gates
- [`results/figures/`](results/figures/): canonical figures
- [`results/tables/claim_scope.json`](results/tables/claim_scope.json): claim boundaries

The default replay is self-contained and CPU-only after setup. It checks compact frozen records
and does not download weather data, load external checkpoints, or retrain the reported models.
Optional weather and checkpoint reruns require the public external assets declared in
`data/manifest.json`. The direct 24-hour Pangu certificate is a static graph result; the empirical
Pangu response records use a 6-hour checkpoint and measure a different object.

## Repository map

- `src/learned_light_cone/`: diagnostics, models, PDE systems, and training code
- `scripts/`: replay, theory checks, and optional experiment reruns
- `configs/` and `data/`: seeds and external-asset declarations
- `results/`: retained tables, figures, and checksums
- `tests/`: software, claim, and reproducibility checks

## License

Released under the [MIT License](LICENSE). External assets remain under their own licenses and are
not redistributed here.
