#!/bin/bash

# Configuration
# NOTE: This script is for training WITH dispersion data.
# Dispersion is NOT added back during compilation since it's already in the training data.
RUN_NAME="01292026"
PROJECT_NAME="aimnet2_ni_pd"
CONFIG_FILE="/expanse/lustre/projects/cwr109/rgao1/AIMNet2mult_train/YAML/train.yaml"
MODEL_FILE="/expanse/lustre/projects/cwr109/rgao1/AIMNet2mult_train/YAML/model.yaml"
DATA_DIR="/expanse/lustre/projects/cwr109/rgao1/AIMNet2mult_train/dataset"
DATASET="${DATA_DIR}/aimnet2_nse_withdispersion.h5"  # Dataset that includes dispersion
SAE_FILE="/expanse/lustre/projects/cwr109/rgao1/AIMNet2mult_train/01292026/sae.yaml"
OUTPUT_DIR="/expanse/lustre/projects/cwr109/rgao1/AIMNet2mult_train/01292026"
SAVE_PATH="${OUTPUT_DIR}/model.pt"
OUTPUT_PREFIX="${OUTPUT_DIR}/compiled/model"

# Dispersion settings: "none" because dispersion is already in training data
DISPERSION="none"

mkdir -p "${OUTPUT_DIR}/compiled"

# Step 1: Compute SAE
python -m aimnet2mult.train.calc_sae "${DATASET}" "${SAE_FILE}"

# Step 2: Train
python -m aimnet2mult.train.cli \
    --config "${CONFIG_FILE}" \
    --model "${MODEL_FILE}" \
    --save "${SAVE_PATH}" \
    run_name="${RUN_NAME}" \
    project_name="${PROJECT_NAME}" \
    data.fidelity_datasets.0="${DATASET}" \
    data.fidelity_weights.0=1.0 \
    data.sae.energy.files.0="${SAE_FILE}"

# Step 3: Compile WITHOUT adding dispersion (since it's already in training data)
python -m aimnet2mult.tools.compile_jit --weights "${SAVE_PATH}" --model "${MODEL_FILE}" --output "${OUTPUT_PREFIX}" --fidelity-level 0 --num-fidelities 1 --sae "${SAE_FILE}" --dispersion "${DISPERSION}"
