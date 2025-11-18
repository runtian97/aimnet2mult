"""Command-line entrypoint for mixed-fidelity AIMNet2 training."""

from __future__ import annotations

import logging

import click

from .configuration import DEFAULT_MODEL_PATH, load_configs
from .runner import run_training


@click.command()
@click.option("--config", type=click.Path(exists=True), default=None, multiple=True, help="Path to extra configuration file(s).")
@click.option("--model", type=click.Path(exists=True), default=str(DEFAULT_MODEL_PATH), help="Path to model definition file.")
@click.option("--load", type=click.Path(exists=True), default=None, help="Path to model weights to load.")
@click.option("--save", type=click.Path(), default=None, help="Path to save final model weights.")
@click.option("--verbose", is_flag=True, help="Enable verbose output.")
@click.argument("args", type=str, nargs=-1)
def main(config, model, load, save, verbose, args):
    """
    Train the mixed-fidelity AIMNet2 model with multi-source data.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("=" * 80)
    logging.info("Mixed-Fidelity AIMNet2 Training")
    logging.info("=" * 80)

    model_cfg, train_cfg = load_configs(config, model, args)
    run_training(model_cfg, train_cfg, load, save)
