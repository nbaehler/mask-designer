"""
Plot aperture example.
"""

from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, ".."))
sys.path.append(CODE_DIR)

import matplotlib.pyplot as plt
import click
from mask_designer.aperture import (
    ApertureOptions,
    rect_aperture,
    line_aperture,
    square_aperture,
    circ_aperture,
)
from slm_controller.hardware import SLMDevices, slm_devices, SLMParam


@click.command()
@click.option(
    "--shape",
    default=ApertureOptions.RECT.value,
    type=click.Choice(ApertureOptions.values()),
    help="Shape of aperture.",
)
@click.option(
    "--n_cells",
    default=10,
    type=int,
    help="Side length for 'square', length for 'line', radius for 'circ'. To set shape for "
    "'rect', use`rect_shape`.",
)
@click.option(
    "--rect_shape",
    default=None,
    nargs=2,
    type=int,
    help="Shape for 'rect' in number of cells; `shape` must be set to 'rect'.",
)
@click.option(
    "--vertical",
    is_flag=True,
    help="Whether line should be vertical (True) or horizontal (False).",
)
@click.option(
    "--show_tick_labels", is_flag=True, help="Whether or not to show cell values along axes.",
)
@click.option(
    "--pixel_pitch",
    default=None,
    nargs=2,
    type=float,
    help="Shape of cell in meters (height, width).",
)
@click.option(
    "--slm_shape",
    default=None,
    nargs=2,
    type=int,
    help="Dimension of SLM in number of cells (height, width).",
)
@click.option(
    "--device",
    type=click.Choice(SLMDevices.values()),
    help="Which device to program with aperture.",
)
def main(
    shape, n_cells, rect_shape, vertical, show_tick_labels, pixel_pitch, slm_shape, device,
):
    """
    Plot SLM aperture.
    """

    if device is None:
        device_config = {
            SLMParam.PIXEL_PITCH: (0.18e-3, 0.18e-3) if pixel_pitch is None else pixel_pitch,
            SLMParam.SLM_SHAPE: (128, 160) if slm_shape is None else slm_shape,
        }
    else:
        device_config = slm_devices[device]

    # create aperture
    ap = None
    if shape == ApertureOptions.RECT.value:
        if rect_shape is None:
            # not provided
            rect_shape = (n_cells, n_cells)
        print(f"Shape : {rect_shape}")
        apert_dim = (
            rect_shape[0] * device_config[SLMParam.PIXEL_PITCH][0],
            rect_shape[1] * device_config[SLMParam.PIXEL_PITCH][1],
        )
        ap = rect_aperture(
            apert_dim=apert_dim,
            slm_shape=device_config[SLMParam.SLM_SHAPE],
            pixel_pitch=device_config[SLMParam.PIXEL_PITCH],
        )
    elif shape == ApertureOptions.LINE.value:
        print(f"Length : {n_cells}")
        length = (
            n_cells * device_config[SLMParam.PIXEL_PITCH][0]
            if vertical
            else n_cells * device_config[SLMParam.PIXEL_PITCH][1]
        )
        ap = line_aperture(
            length=length,
            vertical=vertical,
            slm_shape=device_config[SLMParam.SLM_SHAPE],
            pixel_pitch=device_config[SLMParam.PIXEL_PITCH],
        )
    elif shape == ApertureOptions.SQUARE.value:
        print(f"Side length : {n_cells}")
        ap = square_aperture(
            side=n_cells * device_config[SLMParam.PIXEL_PITCH][0],
            slm_shape=device_config[SLMParam.SLM_SHAPE],
            pixel_pitch=device_config[SLMParam.PIXEL_PITCH],
        )
    elif shape == ApertureOptions.CIRC.value:
        print(f"Radius : {n_cells}")
        ap = circ_aperture(
            radius=n_cells * device_config[SLMParam.PIXEL_PITCH][0],
            slm_shape=device_config[SLMParam.SLM_SHAPE],
            pixel_pitch=device_config[SLMParam.PIXEL_PITCH],
        )

    assert ap is not None

    # plot
    ap.plot(show_tick_labels=show_tick_labels)
    plt.show()


if __name__ == "__main__":
    main()
