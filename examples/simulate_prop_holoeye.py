"""
Simulated propagation of the slm pattern generated using the holoeye software.
"""

from slm_designer.experimental_setup import (
    PhysicalParams,
    physical_params,
    slm_device,
)

from slm_designer.utils import load_holoeye_slm_pattern, show_plot
from slm_designer.simulate_prop import (
    holoeye_fraunhofer,
    neural_holography_asm,
)
from slm_designer.transform_fields import lens_to_lensless

from slm_controller.hardware import (
    SLMParam,
    slm_devices,
)


def simulate_prop_holoeye():
    # Define parameters
    prop_dist = physical_params[PhysicalParams.PROPAGATION_DISTANCE]
    wavelength = physical_params[PhysicalParams.WAVELENGTH]
    slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]
    pixel_pitch = slm_devices[slm_device][SLMParam.PIXEL_PITCH]

    # Load slm phase map computed with holoeye software
    holoeye_slm_field = load_holoeye_slm_pattern()

    # Make it compliant with the data structure used in the project
    slm_field = holoeye_slm_field[0, 0, :, :]

    # Simulate the propagation in the lens setting and show the results
    propped_slm_field = holoeye_fraunhofer(holoeye_slm_field)[0, 0, :, :]
    show_plot(slm_field, propped_slm_field, "Holoeye with lens")

    # Transform the initial phase map to the lensless setting
    holoeye_slm_field = lens_to_lensless(
        holoeye_slm_field, prop_dist, wavelength, slm_shape, pixel_pitch
    )
    slm_field = holoeye_slm_field[0, 0, :, :]

    # Simulate the propagation in the lensless setting and show the results
    propped_slm_field = neural_holography_asm(
        holoeye_slm_field, prop_dist, wavelength, pixel_pitch,
    )[0, 0, :, :]
    show_plot(slm_field, propped_slm_field, "Holoeye without lens")


if __name__ == "__main__":
    simulate_prop_holoeye()
