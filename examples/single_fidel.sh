#!/bin/bash

# Configuration
RUN_NAME="test_1"
PROJECT_NAME="test"
CONFIG_FILE="/projects/bbjt/rgao1/AIMNet2mult/YAML/train.yaml"
MODEL_FILE="/projects/bbjt/rgao1/AIMNet2mult/YAML/model.yaml"
DATA_DIR="/projects/bbjt/rgao1/AIMNet2mult/dataset"
DATASET="${DATA_DIR}/aimnet2_nse_nodispersion.h5"
SAE_FILE="${DATA_DIR}/sae.yaml"
OUTPUT_DIR="/projects/bbjt/rgao1/AIMNet2mult/test"
SAVE_PATH="${OUTPUT_DIR}/model.pt"
OUTPUT_PREFIX="${OUTPUT_DIR}/compiled/model"

# Dispersion settings (none, d3bj, d4)
DISPERSION="d3bj"
DISPERSION_FUNCTIONAL="wb97m"

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

# Step 3: Compile with dispersion
python -m aimnet2mult.tools.compile_jit --weights "${SAVE_PATH}" --model "${MODEL_FILE}" --output "${OUTPUT_PREFIX}" --fidelity-level 0 --num-fidelities 1 --sae "${SAE_FILE}" --dispersion "${DISPERSION}" --dispersion-functional "${DISPERSION_FUNCTIONAL}"
