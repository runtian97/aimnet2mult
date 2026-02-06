#!/bin/bash

# Configuration
RUN_NAME="02012026"
PROJECT_NAME="aimnet2_ni_pd_mult"
CONFIG_FILE="/expanse/lustre/projects/cwr109/rgao1/AIMNet2mult_train/YAML/train.yaml"
MODEL_FILE="/expanse/lustre/projects/cwr109/rgao1/AIMNet2mult_train/YAML/model.yaml"
DATA_DIR="/expanse/lustre/projects/cwr109/rgao1/AIMNet2mult_train/dataset" # Set this to your data directory
FIDELITY_0_DATASET="${DATA_DIR}/aimnet2_nse_nodispersion.h5"
FIDELITY_1_DATASET="${DATA_DIR}/omol_25_nbo_only_no_disp.h5"

FIDELITY_0_WEIGHT=0.2
FIDELITY_1_WEIGHT=1

SAE_FID0="/expanse/lustre/projects/cwr109/rgao1/AIMNet2mult_train/01292026_mult_fid/sae_fid0.yaml"
SAE_FID1="/expanse/lustre/projects/cwr109/rgao1/AIMNet2mult_train/01292026_mult_fid/sae_fid1.yaml"

OUTPUT_DIR="/expanse/lustre/projects/cwr109/rgao1/AIMNet2mult_train/02012026_mult_fid"
PRETRAINED_MODEL="${OUTPUT_DIR}/01292026_model_105_val_loss=-0.0489.pt"
SAVE_PATH="${OUTPUT_DIR}/model.pt"
OUTPUT_PREFIX="${OUTPUT_DIR}/compiled/model"
NUM_FIDELITIES=2
FIDELITY_OFFSET=200
USE_FIDELITY_READOUTS=true

# Dispersion settings per fidelity (none, d3bj, d4)
DISP_FID0="d3bj"
DISP_FID1="d3bj"
FUNC_FID0="wb97m"
FUNC_FID1="wb97m"


mkdir -p "${OUTPUT_DIR}/compiled"

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
    data.fidelity_weights.0="${FIDELITY_0_WEIGHT}" \
    data.fidelity_weights.1="${FIDELITY_1_WEIGHT}" \
    data.sae.energy.files.0="${SAE_FID0}" \
    data.sae.energy.files.1="${SAE_FID1}"

# Compile with dispersion
python -m aimnet2mult.tools.compile_jit --weights "${SAVE_PATH}" --model "${MODEL_FILE}" --output "${OUTPUT_PREFIX}" --fidelity-level 0 --fidelity-offset "${FIDELITY_OFFSET}" --num-fidelities "${NUM_FIDELITIES}" --use-fidelity-readouts "${USE_FIDELITY_READOUTS}" --sae "${SAE_FID0}" --dispersion "${DISP_FID0}" --dispersion-functional "${FUNC_FID0}"
python -m aimnet2mult.tools.compile_jit --weights "${SAVE_PATH}" --model "${MODEL_FILE}" --output "${OUTPUT_PREFIX}" --fidelity-level 1 --fidelity-offset "${FIDELITY_OFFSET}" --num-fidelities "${NUM_FIDELITIES}" --use-fidelity-readouts "${USE_FIDELITY_READOUTS}" --sae "${SAE_FID1}" --dispersion "${DISP_FID1}" --dispersion-functional "${FUNC_FID1}"
