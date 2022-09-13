"""
Script that runs the CITL evaluations of Neural Holography.
"""
import sys
from os.path import abspath, dirname, join

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)

import click
from mask_designer.wrapper import eval_model


@click.command()
@click.option(
    "--channel", type=int, default=1, help="red:0, green:1, blue:2, rgb:3", show_default=True,
)
@click.option(
    "--prop_model",
    type=str,
    default="ASM",
    help="Type of propagation model for reconstruction: ASM / MODEL / PHYSICAL",
    show_default=True,
)
@click.option(
    "--test_phases_path",
    type=str,
    default="./citl/data/test_phases",
    help="Directory where test phases are being stored.",
    show_default=True,
)
@click.option(
    "--test_target_amps_path",
    type=str,
    default="./citl/data/test_target_amps",
    help="Directory where test target amplitudes are being stored.",
    show_default=True,
)
@click.option(
    "--prop_model_dir",
    type=str,
    default="./citl/calibrated_models",
    help="Directory for the CITL-calibrated wave propagation models",
    show_default=True,
)
@click.option(
    "--calibration_path",
    type=str,
    default="./citl/calibration",
    help="Directory where calibration phases are being stored.",
    show_default=True,
)
def main(
    channel, prop_model, test_phases_path, test_target_amps_path, prop_model_dir, calibration_path,
):
    eval_model(
        channel,
        prop_model,
        test_phases_path,
        test_target_amps_path,
        prop_model_dir,
        calibration_path,
    )


if __name__ == "__main__":
    main()
