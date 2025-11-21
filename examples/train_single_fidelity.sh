#!/bin/bash

# Train from scratch with single fidelity (just 1 dataset)
# Same as train.sh but with only one fidelity

# Wandb configuration
RUN_NAME="single_fidelity_001"
PROJECT_NAME="aimnet2_mixed_fidelity"

# Configuration paths
CONFIG_FILE="/Users/nickgao/Desktop/pythonProject/aimnet2mult/examples/YAML/train.yaml"
MODEL_FILE="/Users/nickgao/Desktop/pythonProject/aimnet2mult/examples/YAML/model.yaml"
FIDELITY_0_DATASET="/Users/nickgao/Desktop/pythonProject/aimnet2mult/examples/fake_dataset/fidelity0.h5"
FIDELITY_0_WEIGHT=1.0
SAE_DIR="/Users/nickgao/Desktop/pythonProject/aimnet2mult/examples/fake_dataset"
OUTPUT_DIR="/Users/nickgao/Desktop/pythonProject/aimnet2mult/examples/run"
SAVE_PATH="${OUTPUT_DIR}/model_single_fidelity.pt"
OUTPUT_PREFIX="${OUTPUT_DIR}/model_single_fidelity"
NUM_FIDELITIES=1
FIDELITY_OFFSET=200
USE_FIDELITY_READOUTS=True

mkdir -p "${SAE_DIR}" "${OUTPUT_DIR}"

SAE_FID0="${SAE_DIR}/sae_fid0.yaml"

# Step 1: Compute SAE
python -m aimnet2mult.train.calc_sae "${FIDELITY_0_DATASET}" "${SAE_FID0}"

# Step 2: Train
python -m aimnet2mult.train.cli \
    --config "${CONFIG_FILE}" \
    --model "${MODEL_FILE}" \
    --save "${SAVE_PATH}" \
    run_name="${RUN_NAME}" \
    project_name="${PROJECT_NAME}" \
    data.fidelity_datasets.0="${FIDELITY_0_DATASET}" \
    data.fidelity_weights.0="${FIDELITY_0_WEIGHT}" \
    data.sae.energy.files.0="${SAE_FID0}"

# Step 3: Compile
python -m aimnet2mult.tools.compile_jit --weights "${SAVE_PATH}" --model "${MODEL_FILE}" --output "${OUTPUT_PREFIX}" --fidelity-level 0 --fidelity-offset "${FIDELITY_OFFSET}" --num-fidelities "${NUM_FIDELITIES}" --use-fidelity-readouts "${USE_FIDELITY_READOUTS}" --sae "${SAE_FID0}"
