#!/bin/bash

# Configuration
RUN_NAME="aimnet2mult_continued"
PROJECT_NAME="aimnet2mult"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config/train.yaml"
MODEL_FILE="${SCRIPT_DIR}/config/model.yaml"
DATA_DIR="${SCRIPT_DIR}/data"  # Set this to your data directory
FIDELITY_0_DATASET="${DATA_DIR}/fidelity_0.h5"
FIDELITY_1_DATASET="${DATA_DIR}/fidelity_1.h5"
FIDELITY_2_DATASET="${DATA_DIR}/fidelity_2.h5"
FIDELITY_0_WEIGHT=1.0
FIDELITY_1_WEIGHT=1.0
FIDELITY_2_WEIGHT=1.0
SAE_FID0="${DATA_DIR}/sae_fid0.yaml"
SAE_FID1="${DATA_DIR}/sae_fid1.yaml"
SAE_FID2="${DATA_DIR}/sae_fid2.yaml"
OUTPUT_DIR="${SCRIPT_DIR}/output"
PRETRAINED_MODEL="${OUTPUT_DIR}/model.pt"
SAVE_PATH="${OUTPUT_DIR}/model_continued.pt"
OUTPUT_PREFIX="${OUTPUT_DIR}/compiled_continued/model"
NUM_FIDELITIES=3
FIDELITY_OFFSET=200
USE_FIDELITY_READOUTS=true

# Dispersion settings per fidelity (none, d3bj, d4)
DISP_FID0="d3bj"
DISP_FID1="none"
DISP_FID2="d3bj"
FUNC_FID0="wb97m"
FUNC_FID1=""
FUNC_FID2="wb97m"

mkdir -p "${OUTPUT_DIR}/compiled_continued"

# Continue training with pretrained weights
python -m aimnet2mult.train.cli \
    --config "${CONFIG_FILE}" \
    --model "${MODEL_FILE}" \
    --load "${PRETRAINED_MODEL}" \
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

# Compile with dispersion
python -m aimnet2mult.tools.compile_jit --weights "${SAVE_PATH}" --model "${MODEL_FILE}" --output "${OUTPUT_PREFIX}" --fidelity-level 0 --fidelity-offset "${FIDELITY_OFFSET}" --num-fidelities "${NUM_FIDELITIES}" --use-fidelity-readouts "${USE_FIDELITY_READOUTS}" --sae "${SAE_FID0}" --dispersion "${DISP_FID0}" --dispersion-functional "${FUNC_FID0}"
python -m aimnet2mult.tools.compile_jit --weights "${SAVE_PATH}" --model "${MODEL_FILE}" --output "${OUTPUT_PREFIX}" --fidelity-level 1 --fidelity-offset "${FIDELITY_OFFSET}" --num-fidelities "${NUM_FIDELITIES}" --use-fidelity-readouts "${USE_FIDELITY_READOUTS}" --sae "${SAE_FID1}" --dispersion "${DISP_FID1}"
python -m aimnet2mult.tools.compile_jit --weights "${SAVE_PATH}" --model "${MODEL_FILE}" --output "${OUTPUT_PREFIX}" --fidelity-level 2 --fidelity-offset "${FIDELITY_OFFSET}" --num-fidelities "${NUM_FIDELITIES}" --use-fidelity-readouts "${USE_FIDELITY_READOUTS}" --sae "${SAE_FID2}" --dispersion "${DISP_FID2}" --dispersion-functional "${FUNC_FID2}"
