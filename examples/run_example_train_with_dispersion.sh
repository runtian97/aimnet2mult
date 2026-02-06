#!/bin/bash

# Configuration
# NOTE: This script is for training WITH dispersion data (multi-fidelity).
# Dispersion is NOT added back during compilation since it's already in the training data.
RUN_NAME="aimnet2mult_02062026"
PROJECT_NAME="aimnet2mult"
CONFIG_FILE="/expanse/lustre/projects/cwr109/rgao1/AIMNet2mult_train/YAML/train_dis.yaml"
MODEL_FILE="/expanse/lustre/projects/cwr109/rgao1/AIMNet2mult_train/YAML/model.yaml"
DATA_DIR="/expanse/lustre/projects/cwr109/rgao1/OMol25"  # Set this to your data directory
FIDELITY_0_DATASET="${DATA_DIR}/OMol25_nbo.h5"  # Dataset that includes dispersion
FIDELITY_1_DATASET="${DATA_DIR}/OMol25_nbo_leftover.h5"   # Dataset that includes dispersion

FIDELITY_0_WEIGHT=0.5
FIDELITY_1_WEIGHT=0.5

SAE_FID0="/expanse/lustre/projects/cwr109/rgao1/AIMNet2mult_train/02062026_mult_fid/sae_fid0.yaml"
SAE_FID1="/expanse/lustre/projects/cwr109/rgao1/AIMNet2mult_train/02062026_mult_fid/sae_fid1.yaml"

OUTPUT_DIR="//expanse/lustre/projects/cwr109/rgao1/AIMNet2mult_train/02062026_mult_fid"
SAVE_PATH="${OUTPUT_DIR}/model.pt"
OUTPUT_PREFIX="${OUTPUT_DIR}/compiled/model"
NUM_FIDELITIES=2
FIDELITY_OFFSET=200
USE_FIDELITY_READOUTS=true

# Dispersion settings: "none" because dispersion is already in training data
DISP_FID0="none"
DISP_FID1="none"


mkdir -p "${OUTPUT_DIR}/compiled"

# Step 1: Compute SAE
python -m aimnet2mult.train.calc_sae "${FIDELITY_0_DATASET}" "${SAE_FID0}"
python -m aimnet2mult.train.calc_sae "${FIDELITY_1_DATASET}" "${SAE_FID1}"


# Step 2: Train
python -m aimnet2mult.train.cli \
    --config "${CONFIG_FILE}" \
    --model "${MODEL_FILE}" \
    --save "${SAVE_PATH}" \
    run_name="${RUN_NAME}" \
    project_name="${PROJECT_NAME}" \
    data.fidelity_datasets.0="${FIDELITY_0_DATASET}" \
    data.fidelity_datasets.1="${FIDELITY_1_DATASET}" \
    data.fidelity_weights.0="${FIDELITY_0_WEIGHT}" \
    data.fidelity_weights.1="${FIDELITY_1_WEIGHT}" \
    data.sae.energy.files.0="${SAE_FID0}" \
    data.sae.energy.files.1="${SAE_FID1}"


# Step 3: Compile WITHOUT adding dispersion (since it's already in training data)
python -m aimnet2mult.tools.compile_jit --weights "${SAVE_PATH}" --model "${MODEL_FILE}" --output "${OUTPUT_PREFIX}" --fidelity-level 0 --fidelity-offset "${FIDELITY_OFFSET}" --num-fidelities "${NUM_FIDELITIES}" --use-fidelity-readouts "${USE_FIDELITY_READOUTS}" --sae "${SAE_FID0}" --dispersion "${DISP_FID0}"
python -m aimnet2mult.tools.compile_jit --weights "${SAVE_PATH}" --model "${MODEL_FILE}" --output "${OUTPUT_PREFIX}" --fidelity-level 1 --fidelity-offset "${FIDELITY_OFFSET}" --num-fidelities "${NUM_FIDELITIES}" --use-fidelity-readouts "${USE_FIDELITY_READOUTS}" --sae "${SAE_FID1}" --dispersion "${DISP_FID1}"
