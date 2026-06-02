# **Stratified Disparity**

> Revealing hidden fairness patterns through stratification.

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg)](https://github.com/RichardLitt/standard-readme)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <img src="assets/stratified-disparity-banner.png" alt="Stratified Disparity banner" width="100%">
</p>

**Stratified Disparity** is a fairness metric for link prediction. Instead of evaluating group disparity only at the aggregate level, it measures how disparity changes after stratifying nodes or edges by structural or attribute-based factors such as degree, community, homophily, or learned representations.

The goal is to detect cases where aggregate fairness metrics hide important within-stratum disparities, including Simpson's-paradox effects.

## Table of Contents

- [Overview](#overview)
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Method](#method)
- [API](#api)
- [Examples](#examples)
- [Repository Structure](#repository-structure)
- [Slides](#slides)
- [Citation](#citation)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository is the official implementation of **Stratified Disparity**.

It provides a reusable Python toolkit for computing Stratified Disparity from link prediction outputs, protected attributes, and stratification variables. The package supports the full analysis workflow, including binning, within-bin disparity computation, Stratified Disparity Curve construction, elbow-point selection, and visualization.

The repository includes:

* A reusable Python implementation of Stratified Disparity.
* Utilities for structural or attribute-based stratification.
* Tools for constructing and visualizing Stratified Disparity Curves.
* Elbow-point selection for identifying informative stratification levels.
* Example scripts for applying the metric to custom networks and link prediction results.


The experiments from the paper **“A Stratified Analysis of Link Prediction Fairness”** are provided separately under the [`experiments/`](experiments/) directory.

If you use this code or results in your work, please cite our paper. See the [Citation](#citation) section below.

## Background

Traditional fairness evaluation in link prediction often compares protected groups after aggregating over the entire graph. However, graph data frequently contains strong structural heterogeneity. For example, two demographic groups may have very different degree distributions, and link prediction performance may vary substantially across degree ranges.

This creates a problem:

> Aggregate evaluation can be misleading in both directions: it may hide subgroup-level disparities, or it may make structurally induced disparities appear as protected-group unfairness.

Stratified Disparity addresses this by asking:

1. What happens if we partition the graph by a relevant factor, such as node degree?
2. How large is the performance gap between groups inside each stratum?
3. How does this disparity evolve as the stratification becomes finer?
4. Where is the elbow point where additional stratification gives diminishing returns?

A common form is:

```math
SD(n) = \sum_{j=1}^{n} p_j \cdot Diff(M_{j,1}, M_{j,2}, \ldots, M_{j,t})
```

where:

- `n` is the number of bins,
- `p_j` is the probability or size weight of bin `j`,
- `M_{j,i}` is the performance of group `i` in bin `j`,
- `Diff` measures the disparity across groups, such as standard deviation or absolute gap.

## Stratified Disparity Workflow

The overall workflow of Stratified Disparity is shown below.

![Stratified Disparity Workflow](figures/sd_workflow.png)

Given a network, a protected attribute, and node-level link prediction performance, Stratified Disparity evaluates fairness by comparing protected groups within comparable structural bins.

The workflow consists of the following steps:

1. Run a link prediction method and compute node-level precision.
2. Stratify nodes by degree or another structural attribute.
3. Compute group-level performance within each structural bin.
4. Measure disparity across protected groups inside each bin.
5. Aggregate bin-level disparities using bin weights.
6. Repeat the process with different numbers of bins to obtain the Stratified Disparity Curve.
7. Select the elbow point as the final Stratified Disparity score.

## Install

Clone the repository:

```bash
git clone https://github.com/Oakwoo/stratified-disparity.git
cd stratified-disparity
```

Create an environment:

```bash
conda create -n stratified-disparity python=3.9
conda activate stratified-disparity
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install as a local package:

```bash
pip install -e .
```

## Usage

### 1. Prepare predictions and metadata

The metric requires model predictions, ground-truth labels, protected group labels, and a stratification variable.

```python
import pandas as pd

from stratified_disparity import StratifiedDisparity

# Example input table
# Each row can represent a candidate edge, node pair, or evaluation instance.
df = pd.DataFrame({
    "y_true": [1, 0, 1, 1, 0],
    "y_score": [0.91, 0.22, 0.76, 0.64, 0.31],
    "group": ["A", "B", "A", "B", "A"],
    "degree": [3, 7, 12, 14, 21],
})
```

### 2. Compute Stratified Disparity

```python
sd = StratifiedDisparity(
    group_col="group",
    stratify_col="degree",
    y_true_col="y_true",
    y_score_col="y_score",
    metric="precision",
    diff="std",
)

curve = sd.compute_curve(df, max_bins=32)
print(curve.head())
```

Example output:

```text
   num_bins  stratified_disparity
0         1                0.0657
1         2                0.0412
2         4                0.0218
3         8                0.0099
4        16                0.0101
```

### 3. Detect the elbow point

```python
elbow = sd.find_elbow(curve)
print(elbow)
```

Example output:

```text
{
  "num_bins": 8,
  "stratified_disparity": 0.0099
}
```

### 4. Plot the curve

```python
ax = sd.plot_curve(curve, elbow=elbow)
ax.figure.savefig("figures/stratified_disparity_curve.png", dpi=300, bbox_inches="tight")
```


## API

### `StratifiedDisparity(...)`

Main class for computing the metric.

```python
StratifiedDisparity(
    group_col: str,
    stratify_col: str,
    y_true_col: str,
    y_score_col: str,
    metric: str = "precision",
    diff: str = "std",
    binning: str = "quantile",
)
```

### `compute_curve(df, max_bins=32)`

Computes the Stratified Disparity curve over multiple bin counts.

```python
curve = sd.compute_curve(df, max_bins=32)
```

### `find_elbow(curve)`

Finds the elbow point of the SD curve.

```python
elbow = sd.find_elbow(curve)
```

### `plot_curve(curve, elbow=None)`

Plots the SD curve and optionally marks the elbow point.

```python
ax = sd.plot_curve(curve, elbow=elbow)
```

## Examples

Run a minimal example:

```bash
python examples/run_degree_stratification.py
```

Run an experiment on a graph dataset:

```bash
python experiments/run_link_prediction_audit.py \
  --dataset bowdoin \
  --predictor jaccard \
  --stratify degree \
  --max-bins 32
```

Generate figures:

```bash
python scripts/plot_sd_curve.py \
  --input results/bowdoin_jaccard_degree.csv \
  --output figures/bowdoin_sd_curve.png
```



## Repository Structure

```text
stratified-disparity/
├── assets/                     # README images and diagrams
├── examples/                   # Minimal runnable examples
├── experiments/                # Experiment scripts
├── figures/                    # Generated figures
├── results/                    # Output tables and curves
├── scripts/                    # Plotting and utility scripts
├── slides/                     # presentation slides
├── stratified_disparity/       # Source code
│   ├── __init__.py
│   ├── metric.py
│   ├── binning.py
│   ├── elbow.py
│   └── plotting.py
├── tests/                      # Unit tests
├── requirements.txt
├── LICENSE
└── README.md
```

## Slides

We provide the slides for the ISCS 2026 oral presentation in the `slides/` directory.

- [ISCS 2026 Oral Presentation Slides (PDF)](slides/ISCS_2026_presentation.pdf)

The slides give a concise overview of the motivation, Stratified Disparity metric, experimental setup, and major findings. They are intended as a high-level summary of the paper and can be used together with the code and reproduced results in this repository.

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{wang2026stratifieddisparity,
  title     = {A Stratified Analysis of Link Prediction Fairness},
  author    = {Wang, Weixiang and Soundarajan, Sucheta},
  booktitle = {Proceedings of the International Symposium on Complex Systems},
  year      = {2026}
}
```

## Maintainers

- [Weixiang Wang](https://github.com/wwang69)

## Contributing

Contributions are welcome. Please open an issue first if you plan to add a major feature, change the metric definition, or introduce a new experiment pipeline.

Recommended contribution workflow:

```bash
git checkout -b feature/my-feature
pytest
git commit -m "Add my feature"
git push origin feature/my-feature
```

Then open a pull request.

## License

[MIT](LICENSE) © Weixiang Wang
