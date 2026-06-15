# **Stratified Disparity**

> Revealing hidden fairness patterns through stratification.

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg)](https://github.com/RichardLitt/standard-readme)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <img src="assets/stratified-disparity-banner.png" alt="Stratified Disparity banner" width="100%">
</p>

**Stratified Disparity** is a fairness metric for link prediction. Instead of evaluating group disparity only at the aggregate level, it measures how disparity changes after stratifying nodes or edges by structural or attribute-based factors such as degree, community, homophily, or learned representations.

The goal is to detect cases where aggregate fairness metrics hide important within-stratum disparities, including Simpson's paradox effects.

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

This repository is the official implementation of **Stratified Disparity**, introduced in our paper **“A Stratified Analysis of Link Prediction Fairness.”**

It provides a reusable Python toolkit for computing Stratified Disparity from link prediction outputs, protected attributes, and stratification variables. The package supports the full analysis workflow, including binning, within-bin disparity computation, Stratified Disparity Curve construction, elbow-point selection, and visualization.

The repository includes:

* A reusable Python implementation of Stratified Disparity.
* Utilities for structural or attribute-based stratification.
* Tools for constructing and visualizing Stratified Disparity Curves.
* Elbow-point selection for identifying informative stratification levels.
* Example scripts for applying the metric to custom networks and link prediction results.


The experiments from the paper are provided separately under the [`experiments/`](experiments/) directory.

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

## Simple Usage: Compute a Single Stratified Disparity Score

```python
import numpy as np
import networkx as nx

from stratified_disparity import stratified_disparity

# Structural attribute used for stratification
def bin_by_degree(G, valid_nodes):
    return [G.degree(n) for n in valid_nodes]

# Disparity metric
def std(values):
    return np.std(values, ddof=1)

SD = stratified_disparity(
    valid_nodes=valid_nodes,
    acc_list=acc_list,
    group_labels=group_labels,
    G=G,
    num_bin=10,
    bins_function=bin_by_degree,
    diff_function=std
)

print("Stratified Disparity:", SD)
```

### Inputs

The method requires:

* `valid_nodes`: nodes included in the evaluation.
* `acc_list`: prediction performance (e.g., accuracy, precision, recall, AUC score) associated with each node.
* `group_labels`: protected attribute associated with each node.
* `G`: input network.
* `num_bin`: number of structural bins.
* `bins_function`: function used to compute the structural attribute for stratification.
* `diff_function`: function used to measures performance disparity inside each bin.

### Custom Structural Attributes

Any node-level structural property can be used for stratification.

Example: clustering coefficient

```python
def bin_by_clustering(G, valid_nodes):
    cc = nx.clustering(G)
    return [cc[n] for n in valid_nodes]

SD = stratified_disparity(
    valid_nodes,
    acc_list,
    group_labels,
    G,
    num_bin=10,
    bins_function=bin_by_clustering,
    diff_function=std
)
```

Example: PageRank

```python
def bin_by_pagerank(G, valid_nodes):
    pr = nx.pagerank(G)
    return [pr[n] for n in valid_nodes]
```

### Custom Disparity Functions

Any function that maps a list of performance values to a scalar disparity score can be used.

Standard deviation:

```python
def std(values):
    return np.std(values, ddof=1)
```

Range:

```python
def value_range(values):
    return max(values) - min(values)
```

Mean absolute deviation:

```python
def mad(values):
    mean = np.mean(values)
    return np.mean(np.abs(values - mean))
```


## Experiment Usage: Repeated Evaluation and Curve Analysis
Use the object-oriented API when the graph structure and valid nodes are fixed, but performance scores change across models, epochs, or random seeds.

The Object-oriented interface precomputes and reuses structural attributes and bin assignments, which avoids recomputing the stratification variable for every evaluation.

### 1. Prepare Node-Level Performance Scores and Protected Attributes


```python
acc_list = [1.0, 0.8, 1.0, 0.94, 0.72, ...]
group_labels = {1: 'male', 2: 'female', 3: 'female', ...}
```
Each value in `acc_list` corresponds to a node in `valid_nodes`. `group_labels` is a dictionary that maps each node to its protected attribute.

### 2. Compute Stratified Disparity

```python
import numpy as np
from stratified_disparity import StratifiedDisparity

def bin_by_degree(G, valid_nodes):
    return [G.degree(n) for n in valid_nodes]

def std(values):
    return np.std(values, ddof=1)

sd = StratifiedDisparity(
    G=G,
    valid_nodes=valid_nodes,
    bins_function=bin_by_degree,
    diff_function=std,
    log=True,
)

curve = sd.compute_curve(
    acc_list=acc_list,
    group_labels=group_labels,
    max_bins=32,
)

print(curve)
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


## API Reference

### `stratified_disparity`

```python
stratified_disparity(
    valid_nodes,
    acc_list,
    group_labels,
    G,
    num_bin,
    bins_function,
    diff_function,
    log=True,
    data_log=True
)
```

Compute Stratified Disparity SD(N) by partitioning nodes into structural bins and measuring performance disparity within each bin.

#### Parameters

| Parameter       | Type             | Description                                                                   |
| --------------- | ---------------- | ----------------------------------------------------------------------------- |
| `valid_nodes`   | `list[int]`      | Indices of nodes included in evaluation.                                                 |
| `acc_list`      | `list[float]`    | Node-level prediction performance values.                                     |
| `group_labels`      | `dictionary`    | Node-level protected attribute.                                     |
| `G`             | `networkx.Graph` | Input graph.                                                                  |
| `num_bin`       | `int`            | Number of structural bins (N).                                                |
| `bins_function` | `callable`       | Function that returns a structural attribute value for each node.             |
| `diff_function` | `callable`       | Function that computes disparity within a bin.                                |
| `log`           | `bool`, optional | Use logarithmic binning if `True`, otherwise linear binning. Default: `True`. |
| `data_log`      | `bool`, optional | Print intermediate statistics and debugging information. Default: `True`.     |

#### Returns

| Type    | Description                       |
| ------- | --------------------------------- |
| `float` | Stratified Disparity score SD(N). |

#### Notes

The computation follows three steps:

1. Compute a structural attribute for each node using `bins_function`.
2. Partition nodes into `N` bins according to the structural attribute.
3. Compute within-bin disparity using `diff_function` and aggregate across bins.

When `N=1`, SD(1) reduces to the aggregate disparity without structural stratification.

Increasing `N` reveals whether observed performance disparities are driven by structural imbalance or persist within structurally similar nodes.


### `StratifiedDisparity(...)`

Main class for computing the metric.

```python
sd = StratifiedDisparity(
    G: networkx.Graph,
    valid_nodes: list[int],
    bins_function: callable,
    diff_function: callable,
    log:bool, optional,
)
```

### `compute_curve(df, group_labels, max_bins=32)`

Computes the Stratified Disparity curve over multiple bin counts.
`group_labels` specifies the protected attribute used for evaluation, either as a node attribute name stored in `G` or as a dictionary that maps each node to its protected attribute.

```python
curve = sd.compute_curve(df, group_labels, max_bins=32)
curve = sd.compute_curve(df, 'gender', max_bins=32)
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



## Repository Structure

```text
stratified-disparity/
├── assets/                     # README images and diagrams
├── examples/                   # Minimal runnable examples
├── experiments/                # Experiment scripts
├── figures/                    # Generated figures
├── results/                    # Output tables and curves
├── scripts/                    # Plotting and utility scripts
├── slides/                     # Presentation slides
├── stratified-disparity-iscs2026.pdf # Paper
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

- [ISCS 2026 Oral Presentation Slides](slides/ISCS_2026_Oral_Presentation_Slides.pdf)

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

- [Weixiang Wang](https://github.com/Oakwoo)

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
