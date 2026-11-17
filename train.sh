#!/usr/bin/env bash
set -euo pipefail
# Current visual-prompt method requires a CLIP ViT backbone.
# Additional command-line options override these defaults.
venv/bin/python federated_main.py \
    --model pFedVCluster \
    --config-file configs/trainers/pFedVClusters/vit_b16.yaml \
    --dataset-config-file configs/datasets/office.yaml \
    --output-dir output/office/pFedVCluster/vit_b16_router_checked \
    "$@"
