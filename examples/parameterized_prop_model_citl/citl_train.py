"""
Script that runs the CITL training of Neural Holography.
"""
from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)

import click
from mask_designer.wrapper import train_model
import datetime


@click.command()
@click.option("--channel", type=int, default=1, help="red:0, green:1, blue:2, rgb:3")
@click.option(
    "--pretrained_path",
    type=str,
    default="",
    help="Path of pretrained checkpoints as a starting point",
)
@click.option(
    "--model_path", type=str, default="./citl/models", help="Directory for saving out checkpoints",
)
@click.option(
    "--phase_path",
    type=str,
    default="./citl/precomputed_phases",
    help="Directory for precalculated phases",
)
@click.option(
    "--calibration_path",
    type=str,
    default="./citl/calibration",
    help="Directory where calibration phases are being stored",
)
@click.option(
    "--train_target_amps_path",
    type=str,
    default="./citl/data/train_target_amps",
    help="Directory where train target amplitudes is stored.",
)
@click.option("--lr_model", type=float, default=3e-3, help="Learning rate for model parameters")
@click.option("--lr_phase", type=float, default=5e-3, help="Learning rate for phase")
@click.option("--num_epochs", type=int, default=15, help="Number of epochs")
@click.option(
    "--batch_size", type=int, default=1, help="Size of minibatch"
)  # TODO adapt batch size
@click.option("--no_step_lr", is_flag=True, help="Use of lr scheduler")  # TODO replaced with flag
# @click.option(
#     "--step_lr", type=str2bool, default=True, help="Use of lr scheduler"
# )
# @click.option("--experiment", type=str, default="", help="Name of the experiment")
def main(  # TODO buggy
    channel,
    pretrained_path,
    model_path,
    phase_path,
    calibration_path,
    train_target_amps_path,
    lr_model,
    lr_phase,
    num_epochs,
    batch_size,
    no_step_lr,
    # experiment,
):

    now = datetime.datetime.now()
    experiment = f"{now.year}_{now.month}_{now.day}_{now.hour}h{now.minute}"

    train_model(
        channel,
        pretrained_path,
        model_path,
        phase_path,
        calibration_path,
        train_target_amps_path,
        lr_model,
        lr_phase,
        num_epochs,
        batch_size,
        not no_step_lr,
        experiment,
    )


if __name__ == "__main__":
    main()
