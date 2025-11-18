#!/bin/bash

# Wandb configuration
RUN_NAME="mixed_fidelity_experiment_001"
PROJECT_NAME="aimnet2_mixed_fidelity"

# Configuration paths
CONFIG_FILE="/Users/nickgao/Desktop/pythonProject/aimnet2mult/examples/mismatched/train.yaml"
MODEL_FILE="/Users/nickgao/Desktop/pythonProject/aimnet2mult/examples/mismatched/model.yaml"
FIDELITY_0_DATASET="/Users/nickgao/Desktop/pythonProject/aimnet2mult/sample_data/mismatched/fidelity0.h5"
FIDELITY_1_DATASET="/Users/nickgao/Desktop/pythonProject/aimnet2mult/sample_data/mismatched/fidelity1.h5"
FIDELITY_2_DATASET="/Users/nickgao/Desktop/pythonProject/aimnet2mult/sample_data/mismatched/fidelity2.h5"
FIDELITY_0_WEIGHT=4.0
FIDELITY_1_WEIGHT=2.0
FIDELITY_2_WEIGHT=1.0
SAE_DIR="/Users/nickgao/Desktop/pythonProject/aimnet2mult/sample_data/mismatched"
OUTPUT_DIR="/Users/nickgao/Desktop/pythonProject/aimnet2mult/examples/run"
SAVE_PATH="${OUTPUT_DIR}/model.pt"
OUTPUT_PREFIX="${OUTPUT_DIR}/model"
NUM_FIDELITIES=3
FIDELITY_OFFSET=100
USE_FIDELITY_READOUTS=True

REPO_DIR="/Users/nickgao/Desktop/pythonProject/aimnet2mult"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

mkdir -p "${SAE_DIR}" "${OUTPUT_DIR}"

SAE_FID0="${SAE_DIR}/sae_fid0.yaml"
SAE_FID1="${SAE_DIR}/sae_fid1.yaml"
SAE_FID2="${SAE_DIR}/sae_fid2.yaml"

# Step 1: Compute SAE
python -m aimnet2mult.tools.compute_sae --dataset "${FIDELITY_0_DATASET}" --output "${SAE_FID0}"
python -m aimnet2mult.tools.compute_sae --dataset "${FIDELITY_1_DATASET}" --output "${SAE_FID1}"
python -m aimnet2mult.tools.compute_sae --dataset "${FIDELITY_2_DATASET}" --output "${SAE_FID2}"

# Step 2: Train
python -m aimnet2mult.train.train_mixed_fidelity \
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

# Step 3: Compile
python -m aimnet2mult.tools.compile_jit --weights "${SAVE_PATH}" --model "${MODEL_FILE}" --output "${OUTPUT_PREFIX}" --fidelity-level 0 --fidelity-offset "${FIDELITY_OFFSET}" --num-fidelities "${NUM_FIDELITIES}" --use-fidelity-readouts "${USE_FIDELITY_READOUTS}" --sae "${SAE_FID0}"
python -m aimnet2mult.tools.compile_jit --weights "${SAVE_PATH}" --model "${MODEL_FILE}" --output "${OUTPUT_PREFIX}" --fidelity-level 1 --fidelity-offset "${FIDELITY_OFFSET}" --num-fidelities "${NUM_FIDELITIES}" --use-fidelity-readouts "${USE_FIDELITY_READOUTS}" --sae "${SAE_FID1}"
python -m aimnet2mult.tools.compile_jit --weights "${SAVE_PATH}" --model "${MODEL_FILE}" --output "${OUTPUT_PREFIX}" --fidelity-level 2 --fidelity-offset "${FIDELITY_OFFSET}" --num-fidelities "${NUM_FIDELITIES}" --use-fidelity-readouts "${USE_FIDELITY_READOUTS}" --sae "${SAE_FID2}"
