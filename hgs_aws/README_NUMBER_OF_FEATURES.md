# HGS large-graph workflow with `--number_of_features`

This version changes the CLI so the user explicitly chooses:

```text
--number_of_features 3
```

or

```text
--number_of_features 10
```

## Behaviour

### `--number_of_features 3`

The preprocessing step writes:

```text
Preposs/<DATASET_TAG>/psdfeature_test.json
Preposs/<DATASET_TAG>/edge_index_test.pkl
Preposs/<DATASET_TAG>/preprocess_metadata_3features.json
```

The decoder runner reads:

```text
psdfeature_test.json
edge_index_test.pkl
```

The Excel output reports:

```text
Number of features = 2
```

per Bharat's convention.

### `--number_of_features 10`

The preprocessing step writes:

```text
Preposs/<DATASET_TAG>/psdfeature_test_10features.json
Preposs/<DATASET_TAG>/edge_index_test.pkl
Preposs/<DATASET_TAG>/preprocess_metadata_10features.json
```

The decoder runner reads:

```text
psdfeature_test_10features.json
edge_index_test.pkl
```

The Excel output reports:

```text
Number of features = 9
```

per Bharat's convention.

## End-to-end Jupyter command: 3-feature model

```python
!python3 ~/AWS_poc/code/run_hgs_large_graph_decoder.py \
  --graph_dataset com-dblp \
  --number_of_features 3 \
  --model_path ~/AWS_poc/trained_models/trained_hgs_model_BHOSLIB_and_DIMACS_epochs_12_penalty_coefficient_0.06_lr_0.0004_3features.pth \
  --trained_on bhoslib_and_dimacs \
  --preprocess_if_missing \
  --preposs_root ~/AWS_poc/data/Preposs \
  --data_root ~/AWS_poc/data/online_graphs \
  --results_root ~/AWS_poc/results \
  --decoder both \
  --num_passes_as_percentage_of_total_nodes 0.05 \
  --candidate_limit 5000 \
  --graph_timeout_sec 21600 \
  --run_label aws_hgs_com_dblp_3features \
  --hourly_rate_usd 0.714 \
  --instance_type c7i.4xlarge
```

## End-to-end Jupyter command: 10-feature model

```python
!python3 ~/AWS_poc/code/run_hgs_large_graph_decoder.py \
  --graph_dataset com-dblp \
  --number_of_features 10 \
  --model_path ~/AWS_poc/trained_models/trained_hgs_model_BHOSLIB_and_DIMACS_epochs_12_penalty_coefficient_0.06_lr_0.0004_10features.pth \
  --trained_on bhoslib_and_dimacs \
  --preprocess_if_missing \
  --preposs_root ~/AWS_poc/data/Preposs \
  --data_root ~/AWS_poc/data/online_graphs \
  --results_root ~/AWS_poc/results \
  --decoder both \
  --num_passes_as_percentage_of_total_nodes 0.05 \
  --candidate_limit 5000 \
  --graph_timeout_sec 21600 \
  --run_label aws_hgs_com_dblp_10features \
  --hourly_rate_usd 0.714 \
  --instance_type c7i.4xlarge
```

## Separate preprocessing only

```bash
python3 ~/AWS_poc/code/build_hgs_large_graph_preposs.py \
  --graph_dataset com-dblp \
  --number_of_features 3 \
  --out_root ~/AWS_poc/data/Preposs \
  --data_root ~/AWS_poc/data/online_graphs \
  --large_graph_mode approx
```

or:

```bash
python3 ~/AWS_poc/code/build_hgs_large_graph_preposs.py \
  --graph_dataset com-dblp \
  --number_of_features 10 \
  --out_root ~/AWS_poc/data/Preposs \
  --data_root ~/AWS_poc/data/online_graphs \
  --large_graph_mode approx
```

## Important note

`--large_graph_mode approx` is the default practical mode for full Com-DBLP. Use `--large_graph_mode exact` only for smaller graphs or extracted subgraphs.
