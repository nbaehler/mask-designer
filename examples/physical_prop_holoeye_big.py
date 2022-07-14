"""
Physical propagation of the slm pattern generated using the holoeye software.
"""

import click
from slm_controller.hardware import SLMDevices
from slm_controller import slm
from slm_designer.utils import load_phase_map, quantize_phase_pattern


@click.command()
@click.option(
    "--slm_show_time", type=float, default=5.0, help="Time to show the pattern on the SLM.",
)
def physical_prop_holoeye(show_time):
    # Load the slm pattern generated using the holoeye software
    holoeye_phase_map = load_phase_map("images/holoeye_logo_big.png").angle()
    holoeye_phase_map = quantize_phase_pattern(holoeye_phase_map)

    # Initialize slm
    s = slm.create_slm(SLMDevices.HOLOEYE_LC_2012.value)
    s.set_show_time(show_time)

    # display
    s.imshow(holoeye_phase_map)


if __name__ == "__main__":
    physical_prop_holoeye()
