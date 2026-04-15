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

The model has one site tensor per sequence position. The first tensor carries `input` and `right`, the intermediate tensors carry `left`, `input`, and `right`, and only the last tensor carries the `output` index. For a sequence of length `N`, the classifier uses `bond_dim = N + 1` and `num_classes = N + 1`. Contracting the network with one encoded spike train produces a score vector. Training uses `abs(score)` inside the loss, and the predicted class is the index with the largest absolute score.

Training and checkpoint selection both use `full.csv`. This is intentional: the experiment is meant to study the fully memorized regime, not held-out generalization.

## Saved artifacts

Each experiment writes to:

`output/processed_data/experiments/<experiment_name>/`

Artifacts:

- `checkpoint_best.pt`
- `history.csv`
- `metrics.yaml`
- `confusion_matrix.csv`

`history.csv` stores `train_*` and `full_*` columns, `metrics.yaml` stores `best_full_loss`, `full_loss`, and `full_accuracy`, and `confusion_matrix.csv` is computed on the full dataset.

The checkpoint stores both the learned weights and the model configuration needed to reconstruct the MPS later.

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

The project does not implement a separate tensor inspector. It reconstructs the trained `tensorkrowch` model, resets any traced contraction byproducts, and delegates visualization to `show_tensor_network` from `tensor-network-visualization`.
