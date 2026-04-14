# Dataset Generation

This project generates synthetic binary spike-train datasets from a single YAML configuration file so that experiments can be reproduced exactly.

## Main idea

Each dataset is defined by:

- a name,
- a task,
- the spike-train length,
- the requested number of samples,
- a random seed,
- split ratios,
- and an optional percentage of supervised label noise.

The generator always works with binary spike trains of fixed length. For a sequence length `L`, the maximum number of unique spike trains is `2^L`.

## Configuration file

The canonical configuration file is `configDatasets.yaml`.

For backwards compatibility, the loader also accepts `configDatsets.yaml` if the canonical file is not present.

Example:

```yaml
defaults:
  split_ratios:
    train: 0.7
    val: 0.15
    test: 0.15
  label_noise_pct: 0.0
  noise_scope: train

datasets:
  - name: length5_count_ones
    task: count_ones
    sequence_length: 5
    dataset_size: 32
    seed: 101
```

## Available tasks

The supported labeling rules are:

- `count_ones`: number of ones in the spike train
- `count_zeros`: number of zeros in the spike train
- `adjacent_ones_score`: number of adjacent `11` pairs, equivalent to adding `L - 1` for each run of ones of length `L`

Examples for `adjacent_ones_score`:

- `1101 -> 1`
- `11101 -> 2`
- `111011 -> 3`

## Dataset size and sampling

The generator supports two valid cases:

1. `dataset_size == 2^L`: the full binary space is generated exhaustively.
2. `dataset_size < 2^L`: a reproducible subset is sampled without replacement.

If `dataset_size > 2^L`, the generator stops with a clear error instead of silently introducing duplicate spike trains.

This matters for the initial reference datasets. With `L = 5`, there are only `32` unique spike trains, so requesting `100000` examples is not reasonable if the goal is to avoid duplicates.

## Splits

Each generated dataset is shuffled deterministically from its seed and split into:

- `train`
- `val`
- `test`

The split is reproducible because the same dataset definition and seed always produce the same partition.

## Supervised label noise

The generator can inject label noise through `label_noise_pct`.

In this first version, the noise is local:

- `0` becomes `1`
- the maximum label becomes `max_label - 1`
- intermediate labels become either `y - 1` or `y + 1` with a reproducible random choice

By default, noise is applied only to the training split with `noise_scope: train`.

Each CSV stores:

- `clean_label`: the label before noise
- `label`: the observed label after noise
- `is_noisy`: whether noise was actually applied

## Output layout

Generated datasets are written to `datasets/generated/<dataset_name>/`.

Each dataset directory contains:

- `full.csv`
- `train.csv`
- `val.csv`
- `test.csv`
- `metadata.yaml`

The metadata file records the effective configuration, split sizes, label distributions, and a summary of noisy labels.

## Commands

Generate datasets:

```bash
python scripts/generate_datasets.py --config configDatasets.yaml
```

Clean Python caches and temporary folders:

```bash
python scripts/clean_workspace.py
```
