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
  bond_dim: 8
  patience: 15
  device: auto
  seed: 42

experiments:
  - name: mps_length5_count_ones
    dataset_name: length5_count_ones
```

## Training command

Train one experiment:

```bash
python scripts/train_mps.py --config configTraining.yaml --experiment mps_length5_count_ones
```

The trainer loads the corresponding dataset from `datasets/generated/<dataset_name>/`, applies the fixed local feature map `0 -> [1, 0]`, `1 -> [0, 1]`, and trains an MPS classifier for multiclass classification.

## Saved artifacts

Each experiment writes to:

`output/processed_data/experiments/<experiment_name>/`

Artifacts:

- `checkpoint_best.pt`
- `history.csv`
- `metrics.yaml`
- `confusion_matrix.csv`

The checkpoint stores both the learned weights and the model configuration needed to reconstruct the MPS later.

## Interactive visualization

Reload a checkpoint and open the tensor-network visualizer:

```bash
python scripts/visualize_mps.py --checkpoint output/processed_data/experiments/mps_length5_count_ones/checkpoint_best.pt
```

For automated checks or headless environments:

```bash
python scripts/visualize_mps.py --checkpoint output/processed_data/experiments/mps_length5_count_ones/checkpoint_best.pt --no-show
```

The project does not implement a separate tensor inspector. It reconstructs the trained `tensorkrowch` model and delegates visualization to `show_tensor_network` from `tensor-network-visualization`.
