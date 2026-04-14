# Experimental Background and Hypothesis

## Background

This experiment studies whether a trained Matrix Product State (MPS) can provide a useful bridge between predictive performance and interpretable symbolic structure when the data are binary spike trains with labels defined by explicit sequence rules.

The main motivation for starting with synthetic datasets is control. If the labeling rule is known exactly, then the learning problem is transparent: the model is not discovering an unknown natural phenomenon, but fitting a rule that is already specified at the sequence level. This makes the setting suitable for interpretability analysis because any structure found inside the trained tensor network can be compared against a known target mechanism.

The data in this project are binary spike trains, not tensor trains. Tensor networks are the modeling formalism used to represent the classifier. In particular, the intended model class is a one-dimensional Matrix Product State (MPS).

## Problem Formulation

Let `x = (x_1, ..., x_N)` be a binary spike train of fixed length `N`, with each `x_i` in `{0, 1}`. The initial experiments consider the exhaustive set of all binary sequences of length `N`.

Three supervised classification or regression-style labeling rules are considered:

### 1. Number of Ones

The label is

`y_ones(x) = sum_i x_i`

This task measures whether the model can represent a global counting rule over ones.

### 2. Number of Zeros

The label is

`y_zeros(x) = N - sum_i x_i`

This is complementary to the first task, but it is still useful as a separate controlled dataset because it changes the semantic meaning of the target while preserving a simple exact rule.

### 3. Adjacent-Ones Score

The third task is based on runs of consecutive ones. If a run has length `L`, it contributes `L - 1` to the label. Equivalently, the label counts how many adjacent one-pairs appear in the full sequence:

`y_adj(x) = sum_{i=1}^{N-1} 1[x_i = 1 and x_{i+1} = 1]`

Examples:

- `1101 -> 1`
- `11101 -> 2`
- `111011 -> 3`

This definition is more precise than saying that the label is given by the "number of consecutive ones." The target is not merely the longest run, nor the number of runs, but the total adjacency score induced by all runs.

## Modeling View

Each bit of the spike train is mapped to a two-dimensional local feature vector:

- `0 -> [1, 0]`
- `1 -> [0, 1]`

The full input is represented as the tensor product of these local vectors across the sequence. This makes the representation compatible with efficient contraction against a one-dimensional MPS.

The implementation is expected to use `PyTorch` together with `tensorkrowch` for training, while TensorNetwork-style visualization tools can be used to inspect the learned tensors after optimization.

## Hypothesis

The working hypothesis is the following:

> When an MPS is trained on binary spike-train datasets whose labels are defined by explicit symbolic sequence rules, the optimized tensors may develop internal structure that is informative enough to support logical or rule-based interpretations of the learned classifier.

This should be understood as a hypothesis about interpretability, not only about predictive accuracy. Good task performance alone is not the main endpoint. The key question is whether the trained tensor network contains parameter patterns that can be meaningfully connected to the rule that generated the labels.

## Interpretation Criteria

Evidence in favor of the hypothesis would include observations such as:

- strong sparsity in one or more learned tensors,
- a small number of tensor entries dominating the representation,
- repeated motifs across sites or bonds,
- local structures that align with rule-like behavior, such as counting or adjacency detection.

Such findings would support the idea that logical tensor-network descriptions may be extracted, approximated, or at least motivated from the trained MPS.

## Limits of the Experiment

This experiment does not prove the hypothesis automatically, even if interpretable patterns are observed. At best, positive findings would provide supporting evidence in a controlled setting.

Likewise, a negative result would not conclusively refute the hypothesis. Failure to observe an interpretable structure could come from several sources, including optimization effects, insufficient visualization methods, inadequate tensor dimensions, or the possibility that the learned representation is distributed in a way that is harder to read directly.

For that reason, the initial outcome should be interpreted as exploratory evidence about when explainable logical structure may or may not emerge inside trained MPS models.
