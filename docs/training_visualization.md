# MPS Training and Visualization

This project can now train a Matrix Product State classifier on one generated dataset at a time and reload the best checkpoint later for interactive visualization.

## Training configuration

Training experiments are defined in `configTraining.yaml`.

The file has two levels:

- `defaults`: shared hyperparameters
- `experiments`: one entry per trainable experiment

Example:

```yaml
defaults:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0
  bond_dim: 6
  output_concentration_penalty_weight: 0.25
  tensor_concentration_penalty_weight: 0.25
  patience: 15
  device: auto
  seed: 42

experiments:
  - name: mps_length5_count_ones
    dataset_name: length5_count_ones
```

For these experiments, set:

- `bond_dim = sequence_length + 1`
- `num_classes = sequence_length + 1`
- `output_concentration_penalty_weight` to control how strongly the output probabilities are pushed toward a single dominant class
- `tensor_concentration_penalty_weight` to control how strongly each effective site tensor is pushed toward a small set of dominant entries

## Training command

Run these commands from the repository root with the `.venv` activated.

Train `count_ones`:

```bash
python scripts/train_mps.py --config configTraining.yaml --experiment mps_length5_count_ones
```

Train `count_zeros`:

```bash
python scripts/train_mps.py --config configTraining.yaml --experiment mps_length5_count_zeros
```

Train `adjacent_ones_score`:

```bash
python scripts/train_mps.py --config configTraining.yaml --experiment mps_length5_adjacent_ones_score
```

The trainer loads the corresponding dataset from `datasets/generated/<dataset_name>/`, applies the fixed local feature map `0 -> [1, 0]`, `1 -> [0, 1]`, and trains a manual `tensorkrowch` MPS classifier for multiclass classification.

The model has one site tensor per sequence position. The first tensor carries `input` and `right`, the intermediate tensors carry `left`, `input`, and `right`, and only the last tensor carries the `output` index. For a sequence of length `N`, the classifier uses `bond_dim = N + 1` and `num_classes = N + 1`. The stored trainable values are raw parameters, while the effective tensors used during the forward contraction are their elementwise squares. This makes the training-time MPS positive and removes hidden sign cancellations.

Contracting the network with one encoded spike train produces a nonnegative score vector. Training uses a composite loss made of multiclass cross-entropy on the direct scores, an output-concentration penalty based on the normalized entropy of the softmax distribution, and a tensor-concentration penalty based on the normalized entropy of each effective site tensor. The predicted class is the index with the largest direct score.

Training and checkpoint selection both use `full.csv`. This is intentional: the experiment is meant to study the fully memorized regime, not held-out generalization.

## Saved artifacts

Each experiment writes to:

`output/processed_data/experiments/<experiment_name>/`

Artifacts:

- `checkpoint_best.pt`
- `history.csv`
- `metrics.yaml`
- `confusion_matrix.csv`

`history.csv` stores `train_*` and `full_*` columns for the total loss, the cross-entropy term, the output-concentration penalty term, the tensor-concentration penalty term, accuracy, target activation, off-target activation, strongest incorrect activation, and target margin. `metrics.yaml` stores the final full-dataset summary, and `confusion_matrix.csv` is computed on the full dataset.

The training script also prints one short log line per epoch and a final summary block in the console.

The checkpoint stores the raw site tensors, the MPS parameterization mode, and the model configuration needed to reconstruct the network later. Legacy checkpoints without an explicit parameterization are reloaded as `direct`.

## Interactive visualization

Reload the `count_ones` checkpoint and open the tensor-network visualizer:

```bash
python scripts/visualize_mps.py --checkpoint output/processed_data/experiments/mps_length5_count_ones/checkpoint_best.pt
```

Reload the `count_zeros` checkpoint:

```bash
python scripts/visualize_mps.py --checkpoint output/processed_data/experiments/mps_length5_count_zeros/checkpoint_best.pt
```

Reload the `adjacent_ones_score` checkpoint:

```bash
python scripts/visualize_mps.py --checkpoint output/processed_data/experiments/mps_length5_adjacent_ones_score/checkpoint_best.pt
```

For automated checks or headless environments, add `--no-show`:

```bash
python scripts/visualize_mps.py --checkpoint output/processed_data/experiments/mps_length5_count_ones/checkpoint_best.pt --no-show
```

The project does not implement a separate tensor inspector. It reconstructs the trained `tensorkrowch` model and delegates visualization to `show_tensor_network` from `tensor-network-visualization`. For positive training checkpoints, the visualization shows the effective squared tensors used by the forward pass.

## Post-training canonicalization

To canonicalize a trained checkpoint and save the result as a separate file:

```bash
python scripts/canonicalize_mps.py --checkpoint output/processed_data/experiments/mps_length5_count_ones/checkpoint_best.pt
```

By default, the command writes `checkpoint_canonical.pt` in the same directory as
the input checkpoint.

The command first checks that the original checkpoint already reaches `1.0`
accuracy on `full.csv`. It then canonicalizes the effective tensor network exactly,
runs the canonicalized model again on the same full dataset, and saves the new
checkpoint only if the accuracy stays at `1.0`.

The saved canonical checkpoint is an analysis artifact with
`parameterization: direct`. This keeps the original training checkpoint in the positive squared
parameterization while still allowing exact canonical forms for inspection.

You can choose the exact factorization used during the sweep:

```bash
python scripts/canonicalize_mps.py --checkpoint output/processed_data/experiments/mps_length5_count_ones/checkpoint_best.pt --mode svd
python scripts/canonicalize_mps.py --checkpoint output/processed_data/experiments/mps_length5_count_ones/checkpoint_best.pt --mode qr
```
