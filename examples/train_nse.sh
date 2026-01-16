#!/bin/bash

# Wandb configuration
RUN_NAME="aimnet2_nse_training"
PROJECT_NAME="aimnet2_nse_training"

# Configuration paths
CONFIG_FILE="/Users/nickgao/Desktop/pythonProject/aimnet2mult/examples/YAML/train.yaml"
MODEL_FILE="/Users/nickgao/Desktop/pythonProject/aimnet2mult/examples/YAML/model.yaml"

# Dataset paths - YOUR REAL DATASETS
FIDELITY_0_DATASET="/Users/nickgao/Desktop/pythonProject/AIMNet2_NSE_train/ani1ccx_clean_ev.h5"
FIDELITY_1_DATASET="/Users/nickgao/Desktop/pythonProject/AIMNet2_NSE_train/omol_25_4M_cleaned.h5"
FIDELITY_2_DATASET="/Users/nickgao/Desktop/pythonProject/AIMNet2_NSE_train/aimnet2_nse_cleaned.h5"

# Fidelity weights
FIDELITY_0_WEIGHT=4.0
FIDELITY_1_WEIGHT=2.0
FIDELITY_2_WEIGHT=1.0

# Output paths
SAE_DIR="/Users/nickgao/Desktop/pythonProject/AIMNet2_NSE_train/sae_files"
OUTPUT_DIR="/Users/nickgao/Desktop/pythonProject/aimnet2_training_run"
SAVE_PATH="${OUTPUT_DIR}/model.pt"
OUTPUT_PREFIX="${OUTPUT_DIR}/model"

# Model configuration
NUM_FIDELITIES=3
FIDELITY_OFFSET=200
USE_FIDELITY_READOUTS=True

# Create directories
mkdir -p "${SAE_DIR}" "${OUTPUT_DIR}"

SAE_FID0="${SAE_DIR}/sae_fid0.yaml"
SAE_FID1="${SAE_DIR}/sae_fid1.yaml"
SAE_FID2="${SAE_DIR}/sae_fid2.yaml"

# Step 1: Compute SAE (skip if already exists)
if [ ! -f "${SAE_FID0}" ]; then
    echo "Computing SAE for dataset 0..."
    python -m aimnet2mult.train.calc_sae "${FIDELITY_0_DATASET}" "${SAE_FID0}"
else
    echo "SAE file ${SAE_FID0} already exists, skipping..."
fi

if [ ! -f "${SAE_FID1}" ]; then
    echo "Computing SAE for dataset 1..."
    python -m aimnet2mult.train.calc_sae "${FIDELITY_1_DATASET}" "${SAE_FID1}"
else
    echo "SAE file ${SAE_FID1} already exists, skipping..."
fi

if [ ! -f "${SAE_FID2}" ]; then
    echo "Computing SAE for dataset 2..."
    python -m aimnet2mult.train.calc_sae "${FIDELITY_2_DATASET}" "${SAE_FID2}"
else
    echo "SAE file ${SAE_FID2} already exists, skipping..."
fi

# Step 2: Train
echo "=== Starting Training ==="
python -m aimnet2mult.train.cli \
    --config "${CONFIG_FILE}" \
    --model "${MODEL_FILE}" \
    --save "${SAVE_PATH}" \
    run_name="${RUN_NAME}" \
    project_name="${PROJECT_NAME}" \
    data.fidelity_datasets.0="${FIDELITY_0_DATASET}" \
    data.fidelity_datasets.1="${FIDELITY_1_DATASET}" \
    data.fidelity_datasets.2="${FIDELITY_2_DATASET}" \
    data.fidelity_weights.0="${FIDELITY_0_WEIGHT}" \
    data.fidelity_weights.1="${FIDELITY_1_WEIGHT}" \
    data.fidelity_weights.2="${FIDELITY_2_WEIGHT}" \
    data.sae.energy.files.0="${SAE_FID0}" \
    data.sae.energy.files.1="${SAE_FID1}" \
    data.sae.energy.files.2="${SAE_FID2}"

# Step 3: Compile (only if training succeeded)
if [ $? -eq 0 ]; then
    echo "=== Compiling Models ==="
    python -m aimnet2mult.tools.compile_jit \
        --weights "${SAVE_PATH}" \
        --model "${MODEL_FILE}" \
        --output "${OUTPUT_PREFIX}" \
        --fidelity-level 0 \
        --fidelity-offset "${FIDELITY_OFFSET}" \
        --num-fidelities "${NUM_FIDELITIES}" \
        --use-fidelity-readouts "${USE_FIDELITY_READOUTS}" \
        --sae "${SAE_FID0}"

    python -m aimnet2mult.tools.compile_jit \
        --weights "${SAVE_PATH}" \
        --model "${MODEL_FILE}" \
        --output "${OUTPUT_PREFIX}" \
        --fidelity-level 1 \
        --fidelity-offset "${FIDELITY_OFFSET}" \
        --num-fidelities "${NUM_FIDELITIES}" \
        --use-fidelity-readouts "${USE_FIDELITY_READOUTS}" \
        --sae "${SAE_FID1}"

    python -m aimnet2mult.tools.compile_jit \
        --weights "${SAVE_PATH}" \
        --model "${MODEL_FILE}" \
        --output "${OUTPUT_PREFIX}" \
        --fidelity-level 2 \
        --fidelity-offset "${FIDELITY_OFFSET}" \
        --num-fidelities "${NUM_FIDELITIES}" \
        --use-fidelity-readouts "${USE_FIDELITY_READOUTS}" \
        --sae "${SAE_FID2}"

    echo "=== Training and Compilation Complete! ==="
else
    echo "=== Training failed, skipping compilation ==="
    exit 1
fi
