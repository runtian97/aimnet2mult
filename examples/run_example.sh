#!/bin/bash

# Configuration
RUN_NAME="aimnet2mult_01202026"
PROJECT_NAME="aimnet2mult"
CONFIG_FILE="/projects/bbjt/rgao1/AIMNet2mult/YAML/train.yaml"
MODEL_FILE="/projects/bbjt/rgao1/AIMNet2mult/YAML/model.yaml"
DATA_DIR="/projects/bbjt/rgao1/AIMNet2mult/dataset"  # Set this to your data directory
FIDELITY_0_DATASET="${DATA_DIR}/aimnet2_nse_nodispersion.h5"
FIDELITY_1_DATASET="${DATA_DIR}/omol_25_4M_nodispersion.h5"
FIDELITY_2_DATASET="${DATA_DIR}/ani1ccx_clean_ev.h5"
FIDELITY_0_WEIGHT=1.0
FIDELITY_1_WEIGHT=1.0
FIDELITY_2_WEIGHT=1.0
SAE_FID0="${DATA_DIR}/sae_fid0.yaml"
SAE_FID1="${DATA_DIR}/sae_fid1.yaml"
SAE_FID2="${DATA_DIR}/sae_fid2.yaml"
OUTPUT_DIR="/projects/bbjt/rgao1/AIMNet2mult/01202026"
SAVE_PATH="${OUTPUT_DIR}/model.pt"
OUTPUT_PREFIX="${OUTPUT_DIR}/compiled/model"
NUM_FIDELITIES=3
FIDELITY_OFFSET=200
USE_FIDELITY_READOUTS=true

# Dispersion settings per fidelity (none, d3bj, d4)
DISP_FID0="d3bj"
DISP_FID1="d3bj"
DISP_FID2="none"
FUNC_FID0="wb97m"
FUNC_FID1="wb97m"
FUNC_FID2=""

mkdir -p "${OUTPUT_DIR}/compiled"

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

# Step 3: Compile with dispersion
python -m aimnet2mult.tools.compile_jit --weights "${SAVE_PATH}" --model "${MODEL_FILE}" --output "${OUTPUT_PREFIX}" --fidelity-level 0 --fidelity-offset "${FIDELITY_OFFSET}" --num-fidelities "${NUM_FIDELITIES}" --use-fidelity-readouts "${USE_FIDELITY_READOUTS}" --sae "${SAE_FID0}" --dispersion "${DISP_FID0}" --dispersion-functional "${FUNC_FID0}"
python -m aimnet2mult.tools.compile_jit --weights "${SAVE_PATH}" --model "${MODEL_FILE}" --output "${OUTPUT_PREFIX}" --fidelity-level 1 --fidelity-offset "${FIDELITY_OFFSET}" --num-fidelities "${NUM_FIDELITIES}" --use-fidelity-readouts "${USE_FIDELITY_READOUTS}" --sae "${SAE_FID1}" --dispersion "${DISP_FID1}"
python -m aimnet2mult.tools.compile_jit --weights "${SAVE_PATH}" --model "${MODEL_FILE}" --output "${OUTPUT_PREFIX}" --fidelity-level 2 --fidelity-offset "${FIDELITY_OFFSET}" --num-fidelities "${NUM_FIDELITIES}" --use-fidelity-readouts "${USE_FIDELITY_READOUTS}" --sae "${SAE_FID2}" --dispersion "${DISP_FID2}" --dispersion-functional "${FUNC_FID2}"
