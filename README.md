# Spike-Train Explainability with Matrix Product States

This repository studies whether a one-dimensional tensor network can learn simple rule-based classifications on binary spike trains and still retain interpretable internal structure after training.

The central question is not only whether an MPS classifier can fit these tasks, but whether the optimized tensors exhibit sparse, dominant, or structured patterns that can later be related to logical tensor-network descriptions.

## Experimental Goal

The initial goal is to train Matrix Product State (MPS) models on synthetic spike-train datasets whose labels are defined by explicit symbolic rules over binary sequences. These controlled tasks make it possible to inspect the trained network in a setting where the target rule is known in advance.

The repository already includes a reproducible dataset-generation pipeline driven by YAML configuration. Training and tensor inspection are the next steps.

## Synthetic Datasets

The first version of the experiment focuses on exhaustive binary strings of fixed length. Each binary string is treated as a spike train and assigned one label according to one of the following rules:

1. `count_ones`: the label is the number of ones in the spike train.
2. `count_zeros`: the label is the number of zeros in the spike train.
3. `adjacent_ones_score`: the label is the sum of contributions from runs of consecutive ones, where a run of length `L` contributes `L - 1`.

For the third dataset:

- `1101 -> 1`
- `11101 -> 2`
- `111011 -> 3`

The third rule is therefore not simply the number of ones or the maximum run length. It is a run-based score that counts how many adjacent one-pairs are present across the full sequence.

## Planned Modeling Pipeline

The intended workflow is:

1. Generate exhaustive fixed-length binary spike trains and assign labels according to one of the three rules above.
2. Encode each bit with a local two-dimensional feature map:
   - `0 -> [1, 0]`
   - `1 -> [0, 1]`
3. Form the full input as a tensor-product state over the sequence positions.
4. Train a one-dimensional MPS classifier on the full dataset using `PyTorch` with `tensorkrowch`.
5. Inspect and visualize the optimized tensor network using TensorNetwork-style visualization tools.

This setup keeps the input map simple and fully discrete, which makes it easier to relate learned tensor entries to symbolic sequence rules.

## Current Repository Capabilities

The project currently includes:

- a typed Python package under `src/`,
- a configurable dataset generator driven by `configDatasets.yaml`,
- a separate training configuration driven by `configTraining.yaml`,
- reproducible `train`, `val`, and `test` splits,
- optional supervised label noise with controlled percentage,
- generated reference datasets for the three rule-based tasks with sequence length `5`,
- dataset metadata stored alongside each generated dataset,
- an MPS training pipeline based on `tensorkrowch` and `PyTorch`,
- checkpoint reload plus interactive tensor-network visualization.

The generator writes datasets to `datasets/generated/<dataset_name>/` as CSV files plus a `metadata.yaml` summary.

## Repository Layout

```text
src/                    Python package for configuration, labeling, generation, and writing
scripts/                Entry-point scripts for generation and cleanup
datasets/generated/     Generated canonical datasets
docs/                   Documentation for dataset generation, training, and visualization
output/                 Reserved for later analysis artifacts
tests/                  Automated tests
```

## Dataset Generation

Use the configured datasets:

```bash
python scripts/generate_datasets.py --config configDatasets.yaml
```

The detailed generation workflow is documented in `docs/dataset_generation.md`.

## MPS Training

Run these commands from the repository root with the `.venv` already activated.

For the current length-5 experiments, set:

- `bond_dim = sequence_length + 1 = 6`
- `num_classes = sequence_length + 1 = 6`
- `one_hot_penalty_weight = 1.0`
- `concentration_penalty_weight = 0.25`

Train `count_ones`:

```bash
python scripts/train_mps.py --config configTraining.yaml --experiment mps_length5_count_ones
python scripts/train_mps.py --config configTraining.yaml --experiment mps_length5_count_zeros
python scripts/train_mps.py --config configTraining.yaml --experiment mps_length5_adjacent_ones_score
```

Training uses `full.csv` both for optimization and for the checkpoint selection criterion, because the purpose of this experiment is to study the fully memorized limit rather than generalization.

The classifier is implemented as a manual `tensorkrowch` tensor network with one site tensor per spike-train position. The first tensor carries only `input` and `right`, the intermediate tensors carry `left`, `input`, and `right`, and only the last tensor carries the class `output` index. The stored trainable values are now raw parameters, but the effective tensors used in the MPS contraction are their elementwise squares. This makes the training-time MPS strictly nonnegative and removes sign cancellations from the forward pass.

After contracting the network with an encoded spike train, the model returns one nonnegative score per category. Training uses a composite objective with three terms: cross-entropy on the direct scores, an MSE penalty that pushes the full output vector toward a one-hot target, and a concentration penalty based on the normalized entropy of each effective site tensor so that the network tends to place most of its mass in a small number of entries. Prediction chooses the category with the largest direct score.

During training, the console now prints one compact line per epoch with the loss split, including the concentration term, together with accuracy, target activation, off-target activation, and target margin. At the end it also prints a final summary over the full dataset. The same metrics are written to `history.csv` and `metrics.yaml`.

## Interactive Visualization

Open the `count_ones` checkpoint:

```bash
python scripts/visualize_mps.py --checkpoint output/processed_data/experiments/mps_length5_count_ones/checkpoint_best.pt
python scripts/visualize_mps.py --checkpoint output/processed_data/experiments/mps_length5_count_zeros/checkpoint_best.pt
python scripts/visualize_mps.py --checkpoint output/processed_data/experiments/mps_length5_adjacent_ones_score/checkpoint_best.pt
```

To inspect the contraction scheme for each sample in the dataset linked to the checkpoint, use `--per-sample`:

```bash
python scripts/visualize_mps.py --checkpoint output/processed_data/experiments/mps_length5_count_ones/checkpoint_best.pt --per-sample
python scripts/visualize_mps.py --checkpoint output/processed_data/experiments/mps_length5_count_zeros/checkpoint_best.pt --per-sample
python scripts/visualize_mps.py --checkpoint output/processed_data/experiments/mps_length5_adjacent_ones_score/checkpoint_best.pt --per-sample
```

For a quick inspection without opening the whole dataset, add `--limit`:

```bash
python scripts/visualize_mps.py --checkpoint output/processed_data/experiments/mps_length5_count_ones/checkpoint_best.pt --per-sample --limit 3
python scripts/visualize_mps.py --checkpoint output/processed_data/experiments/mps_length5_count_zeros/checkpoint_best.pt --per-sample --limit 3
python scripts/visualize_mps.py --checkpoint output/processed_data/experiments/mps_length5_adjacent_ones_score/checkpoint_best.pt --per-sample --limit 3
```

When `--per-sample` is active, the command contracts the network with each spike train from the selected split, opens the contraction view for that sample, and prints the `spike_train`, labels, prediction, and output scores in the console. For training checkpoints, the visualization shows the effective positive tensors used in the forward contraction, not the unsquared raw parameters.

For automated checks or headless environments, add `--no-show` to any of the commands above:

```bash
python scripts/visualize_mps.py --checkpoint output/processed_data/experiments/mps_length5_count_ones/checkpoint_best.pt --no-show
python scripts/visualize_mps.py --checkpoint output/processed_data/experiments/mps_length5_count_ones/checkpoint_best.pt --per-sample --no-show
```

## Post-Training Canonicalization

You can canonicalize a trained checkpoint after training and save the result as a
new checkpoint:

```bash
python scripts/canonicalize_mps.py --checkpoint output/processed_data/experiments/mps_length5_count_ones/checkpoint_best.pt
```

By default, the command writes `checkpoint_canonical.pt` next to the input
checkpoint. Before saving, it reevaluates the canonicalized model on the full
dataset linked to the checkpoint and refuses to save unless the accuracy remains
exactly `1.0`.

The canonical checkpoint is an exact analysis artifact. If the source model uses
the positive squared parameterization, the command first canonicalizes the
effective positive tensors and then saves the canonicalized network as a
`direct` checkpoint. That canonical form may contain signs internally, while the
original training checkpoint remains a positive-MPS checkpoint.

The training and visualization workflow is documented in `docs/training_visualization.md`.

## What Would Count as Supportive Evidence

The working hypothesis is that, after training on these explicit rule-based tasks, the internal tensors of the optimized MPS may exhibit interpretable structure. Examples of supportive evidence would include:

- highly sparse tensors,
- a small subset of dominant entries,
- repeated local patterns across sites,
- parameter configurations that can be related to simple logical rules over the input sequence.

Observing this kind of structure would support the hypothesis that logical explanations may be recoverable from trained tensor networks in this setting. It would not, by itself, constitute a full proof.

## Current Scope

At this stage, the repository is meant to define the experimental direction clearly:

- which datasets will be generated,
- which input representation will be used,
- which class of tensor-network model will be trained,
- and what kind of interpretability signal will be searched for after optimization.

The repository now has the first complete path from synthetic dataset generation to MPS training, checkpointing, and later visualization of the learned tensor network.
