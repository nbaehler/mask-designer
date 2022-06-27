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
    "--root_path",
    type=str,
    default="./phases",
    help="Directory where test phases are being stored.",
)
@click.option(
    "--prop_model_dir",
    type=str,
    default="./calibrated_models",
    help="Directory for the CITL-calibrated wave propagation models",
)
@click.option(
    "--calibration_path",
    type=str,
    default="./calibration",
    help="Directory where calibration phases are being stored.",
)
def citl_eval(channel, prop_model, root_path, prop_model_dir, calibration_path):

    eval(channel, prop_model, root_path, prop_model_dir, calibration_path)


if __name__ == "__main__":
    citl_eval()
