"""
Physical propagation of the slm pattern generated using the holoeye software.
"""

import click
from slm_controller.hardware import SLMDevices
from slm_controller import slm
from slm_designer.utils import load_holoeye_slm_pattern, quantize_phase_pattern


@click.command()
@click.option("--show_time", type=float, default=5.0, help="Time to show the pattern on the SLM.")
def physical_prop_holoeye(show_time):
    # Load the slm pattern generated using the holoeye software
    holoeye_slm_field = load_holoeye_slm_pattern().angle()
    holoeye_slm_field = quantize_phase_pattern(holoeye_slm_field)

    # Initialize slm
    s = slm.create_slm(SLMDevices.HOLOEYE_LC_2012.value)
    s.set_show_time(show_time)

    # display
    s.imshow(holoeye_slm_field)


if __name__ == "__main__":
    physical_prop_holoeye()
