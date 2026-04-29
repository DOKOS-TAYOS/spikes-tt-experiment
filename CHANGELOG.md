# Changelog

## Unreleased

- Replaced the output-side one-hot MSE term with a multiclass output-concentration penalty based on the normalized entropy of the softmax distribution.
- Kept and renamed the internal tensor regularization as `tensor_concentration_penalty`, so the training objective now distinguishes clearly between output concentration and tensor concentration.
- Updated `configTraining.yaml`, saved metrics, training logs, and documentation to use the new penalty names:
  `output_concentration_penalty_weight` and `tensor_concentration_penalty_weight`.
