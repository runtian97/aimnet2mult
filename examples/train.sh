#!/bin/bash

# Wandb configuration
RUN_NAME="aimnet2_mf_v5_sgd"
PROJECT_NAME="aimnet2_training"

# Configuration paths - use V5 SGD to escape local minimum
CONFIG_FILE="/Users/nickgao/Desktop/pythonProject/aimnet2mult_train/train_v5_sgd.yaml"
MODEL_FILE="/Users/nickgao/Desktop/pythonProject/aimnet2mult/examples/YAML/model.yaml"
FIDELITY_0_DATASET="/Users/nickgao/Desktop/pythonProject/AIMNet2_NSE_train/aimnet2_nse_cleaned.h5"
FIDELITY_1_DATASET="/Users/nickgao/Desktop/pythonProject/AIMNet2_NSE_train/ani1ccx_clean_ev.h5"
FIDELITY_2_DATASET="/Users/nickgao/Desktop/pythonProject/AIMNet2_NSE_train/omol_25_4M_cleaned.h5"
FIDELITY_0_WEIGHT=1.0
FIDELITY_1_WEIGHT=1.0
FIDELITY_2_WEIGHT=0.5
SAE_DIR="/Users/nickgao/Desktop/pythonProject/aimnet2mult_train"
OUTPUT_DIR="/Users/nickgao/Desktop/pythonProject/aimnet2mult_train/output"
SAVE_PATH="${OUTPUT_DIR}/model.pt"
OUTPUT_PREFIX="${OUTPUT_DIR}/model"
NUM_FIDELITIES=3
FIDELITY_OFFSET=200
USE_FIDELITY_READOUTS=True

mkdir -p "${SAE_DIR}" "${OUTPUT_DIR}"

SAE_FID0="${SAE_DIR}/sae_fid0.yaml"
SAE_FID1="${SAE_DIR}/sae_fid1.yaml"
SAE_FID2="${SAE_DIR}/sae_fid2.yaml"

# Step 1: Compute SAE
python -m aimnet2mult.train.calc_sae "${FIDELITY_0_DATASET}" "${SAE_FID0}"
python -m aimnet2mult.train.calc_sae "${FIDELITY_1_DATASET}" "${SAE_FID1}"
python -m aimnet2mult.train.calc_sae "${FIDELITY_2_DATASET}" "${SAE_FID2}"

# Step 2: Train
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

# Step 3: Compile
python -m aimnet2mult.tools.compile_jit --weights "${SAVE_PATH}" --model "${MODEL_FILE}" --output "${OUTPUT_PREFIX}" --fidelity-level 0 --fidelity-offset "${FIDELITY_OFFSET}" --num-fidelities "${NUM_FIDELITIES}" --use-fidelity-readouts "${USE_FIDELITY_READOUTS}" --sae "${SAE_FID0}"
python -m aimnet2mult.tools.compile_jit --weights "${SAVE_PATH}" --model "${MODEL_FILE}" --output "${OUTPUT_PREFIX}" --fidelity-level 1 --fidelity-offset "${FIDELITY_OFFSET}" --num-fidelities "${NUM_FIDELITIES}" --use-fidelity-readouts "${USE_FIDELITY_READOUTS}" --sae "${SAE_FID1}"
python -m aimnet2mult.tools.compile_jit --weights "${SAVE_PATH}" --model "${MODEL_FILE}" --output "${OUTPUT_PREFIX}" --fidelity-level 2 --fidelity-offset "${FIDELITY_OFFSET}" --num-fidelities "${NUM_FIDELITIES}" --use-fidelity-readouts "${USE_FIDELITY_READOUTS}" --sae "${SAE_FID2}"
