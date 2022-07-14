"""
Script that runs the CITL evaluations of Neural Holography. #TODO not working entirely
"""

import click
from slm_designer.wrapper import eval


@click.command()
@click.option("--channel", type=int, default=1, help="red:0, green:1, blue:2, rgb:3")
@click.option(
    "--prop_model",
    type=str,
    default="ASM",
    help="Type of propagation model for reconstruction: ASM / MODEL / CAMERA",
)
@click.option(
    "--test_phases_path",
    type=str,
    default="./citl/data/test_phases",
    help="Directory where test phases are being stored.",
)
@click.option(
    "--test_target_amps_path",
    type=str,
    default="./citl/data/test_target_amps",
    help="Directory where test target amplitudes are being stored.",
)
@click.option(
    "--prop_model_dir",
    type=str,
    default="./citl/calibrated_models",  # TODO normally calibrated in manual step? For now just copy there by hand ...
    help="Directory for the CITL-calibrated wave propagation models",
)
@click.option(
    "--calibration_path",
    type=str,
    default="./citl/calibration",
    help="Directory where calibration phases are being stored.",
)
def parameterized_prop_model_citl_eval(
    channel, prop_model, test_phases_path, test_target_amps_path, prop_model_dir, calibration_path,
):
    eval(
        channel,
        prop_model,
        test_phases_path,
        test_target_amps_path,
        prop_model_dir,
        calibration_path,
    )


if __name__ == "__main__":
    parameterized_prop_model_citl_eval()
