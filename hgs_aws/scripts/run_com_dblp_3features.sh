#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$HOME/AWS_poc/results" "$HOME/AWS_poc/data/Preposs" "$HOME/AWS_poc/trained_models"
source "$HOME/venvs/hgs-aws/bin/activate"

python3 "$HOME/AWS_poc/code/run_hgs_large_graph_decoder.py" \
  --graph_dataset com-dblp \
  --number_of_features 3 \
  --model_path "$HOME/AWS_poc/trained_models/trained_hgs_model_BHOSLIB_and_DIMACS_epochs_12_penalty_coefficient_0.06_lr_0.0004_3features.pth" \
  --trained_on bhoslib_and_dimacs \
  --preprocess_if_missing \
  --preposs_root "$HOME/AWS_poc/data/Preposs" \
  --data_root "$HOME/AWS_poc/data/online_graphs" \
  --results_root "$HOME/AWS_poc/results" \
  --decoder both \
  --num_passes_as_percentage_of_total_nodes 0.05 \
  --candidate_limit 5000 \
  --graph_timeout_sec 21600 \
  --run_label aws_hgs_com_dblp_3features \
  --hourly_rate_usd 0.714 \
  --instance_type c7i.4xlarge
