"""
Utility functions for fidelity-specific configuration validation.

This module provides validation and configuration utilities for
mixed-fidelity training.
"""

import logging
from omegaconf import OmegaConf


def validate_fidelity_configuration(train_cfg):
    """
    Validate that the fidelity configuration is properly structured.

    Args:
        train_cfg: OmegaConf configuration object

    Raises:
        ValueError: If configuration is invalid
    """
    # Check that fidelity_datasets exists
    if not hasattr(train_cfg.data, 'fidelity_datasets'):
        raise ValueError("Configuration must specify 'data.fidelity_datasets'")

    fidelity_datasets = train_cfg.data.fidelity_datasets
    if not fidelity_datasets:
        raise ValueError("fidelity_datasets cannot be empty")

    # Get number of fidelities
    num_fidelities = len(fidelity_datasets)
    logging.info(f"Configuration has {num_fidelities} fidelity level(s)")

    # Validate fidelity keys are consecutive integers starting from 0
    expected_keys = set(range(num_fidelities))
    actual_keys = set(int(k) for k in fidelity_datasets.keys())

    if expected_keys != actual_keys:
        raise ValueError(
            f"Fidelity dataset keys must be consecutive integers starting from 0. "
            f"Expected: {sorted(expected_keys)}, Got: {sorted(actual_keys)}"
        )

    # Validate fidelity_weights if present
    if hasattr(train_cfg.data, 'fidelity_weights') and train_cfg.data.fidelity_weights is not None:
        weight_keys = set(int(k) for k in train_cfg.data.fidelity_weights.keys())
        if weight_keys != actual_keys:
            raise ValueError(
                f"Fidelity weights must be specified for all fidelity levels. "
                f"Expected keys: {sorted(actual_keys)}, Got: {sorted(weight_keys)}"
            )

        # Check that all weights are positive
        for fid, weight in train_cfg.data.fidelity_weights.items():
            if weight <= 0:
                raise ValueError(f"Fidelity weight for level {fid} must be positive, got {weight}")

        logging.info(f"Using weighted sampling: {dict(train_cfg.data.fidelity_weights)}")

    # Validate SAE configuration if present
    if hasattr(train_cfg.data, 'sae') and train_cfg.data.sae is not None:
        if hasattr(train_cfg.data.sae, 'energy') and train_cfg.data.sae.energy is not None:
            sae_energy = train_cfg.data.sae.energy
            if hasattr(sae_energy, 'files') and sae_energy.files is not None:
                sae_keys = set(int(k) for k in sae_energy.files.keys())
                if sae_keys != actual_keys:
                    raise ValueError(
                        f"SAE files must be specified for all fidelity levels. "
                        f"Expected keys: {sorted(actual_keys)}, Got: {sorted(sae_keys)}"
                    )
                logging.info(f"SAE corrections configured for all {num_fidelities} fidelity levels")

    # Validate fidelity offset if present
    if hasattr(train_cfg, 'fidelity_offset'):
        offset = train_cfg.fidelity_offset
        use_base_model = getattr(train_cfg, 'use_base_model', False)
        # Allow offset=0 for single fidelity with base model (aimnet2 compat mode)
        if offset < 0:
            raise ValueError(f"fidelity_offset must be non-negative, got {offset}")
        if offset == 0 and num_fidelities > 1:
            raise ValueError(f"fidelity_offset must be positive for multi-fidelity training ({num_fidelities} fidelities)")
        if offset == 0:
            logging.info(f"Using fidelity_offset: 0 (single-fidelity / aimnet2 compat mode)")
        else:
            logging.info(f"Using fidelity_offset: {offset}")

    # Validate use_fidelity_readouts if present
    if hasattr(train_cfg, 'use_fidelity_readouts'):
        use_fidelity_readouts = train_cfg.use_fidelity_readouts
        logging.info(f"Using fidelity-specific readouts: {use_fidelity_readouts}")

    logging.info("✓ Fidelity configuration validated successfully")
