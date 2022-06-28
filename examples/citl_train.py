import click
from slm_designer.wrapper import train_model, str2bool

from slm_designer.experimental_setup import (
    PhysicalParams,
    physical_params,
    slm_device,
    cam_device,
)


@click.command()
@click.option("--channel", type=int, default=1, help="red:0, green:1, blue:2, rgb:3")
@click.option(
    "--pretrained_path",
    type=str,
    default="",
    help="Path of pretrained checkpoints as a starting point",
)
@click.option(
    "--model_path", type=str, default="./models", help="Directory for saving out checkpoints",
)
@click.option(
    "--phase_path",
    type=str,
    default="./precomputed_phases",
    help="Directory for precalculated phases",
)
@click.option(
    "--calibration_path",
    type=str,
    default="./calibration",
    help="Directory where calibration phases are being stored",
)
@click.option(
    "--train_data_path",
    type=str,
    default="./data/train",
    help="Directory where train data is stored.",
)
@click.option("--lr_model", type=float, default=3e-3, help="Learning rate for model parameters")
@click.option("--lr_phase", type=float, default=5e-3, help="Learning rate for phase")
@click.option("--num_epochs", type=int, default=15, help="Number of epochs")
@click.option("--batch_size", type=int, default=2, help="Size of minibatch")
@click.option("--step_lr", type=str2bool, default=True, help="Use of lr scheduler")
@click.option("--experiment", type=str, default="", help="Name of the experiment")
def citl_train(
    channel,
    pretrained_path,
    model_path,
    phase_path,
    calibration_path,
    train_data_path,
    lr_model,
    lr_phase,
    num_epochs,
    batch_size,
    step_lr,
    experiment,
):
    slm_settle_time = 0.5  # TODO set those in click
    prop_dist = physical_params[PhysicalParams.PROPAGATION_DISTANCE]
    wavelength = physical_params[PhysicalParams.WAVELENGTH]

    train_model(
        slm_device,
        cam_device,
        slm_settle_time,
        channel,
        pretrained_path,
        model_path,
        phase_path,
        calibration_path,
        train_data_path,
        lr_model,
        lr_phase,
        num_epochs,
        batch_size,
        step_lr,
        experiment,
        prop_dist,
        wavelength,
    )


if __name__ == "__main__":
    citl_train()
