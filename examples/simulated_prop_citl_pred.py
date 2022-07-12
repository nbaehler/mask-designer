"""
Simulated propagation of the slm pattern generated using the holoeye software.
"""

from slm_designer.experimental_setup import (
    Params,
    params,
    slm_device,
)
from slm_designer.simulated_prop import simulated_prop

from slm_designer.utils import (
    load_phase_map,
    pad_tensor_to_shape,
    show_plot,
)
from slm_designer.propagation import (
    holoeye_fraunhofer,
    neural_holography_asm,
)
from slm_designer.transform_phase_maps import transform_to_neural_holography_setting

from slm_controller.hardware import SLMParam, slm_devices


def simulated_prop_citl_pred():
    # Define parameters
    prop_dist = params[Params.PROPAGATION_DISTANCE]
    wavelength = params[Params.WAVELENGTH]
    pixel_pitch = slm_devices[slm_device][SLMParam.PIXEL_PITCH]
    slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]

    # Load slm phase map computed with CITL
    holoeye_phase_map = load_phase_map("citl/predictions/0_holoeye_logo_pred_phases_ASM_green.png")

    # Pad roi to full slm shape
    holoeye_phase_map = pad_tensor_to_shape(
        holoeye_phase_map, slm_shape
    )  # TODO padding really needed? Done in citl_pred as well, we'll see
    unpacked_phase_map = holoeye_phase_map[0, 0, :, :]

    # Simulate the propagation in the lens setting and show the results
    propped_phase_map = simulated_prop(holoeye_phase_map, holoeye_fraunhofer)
    show_plot(unpacked_phase_map, propped_phase_map, "CITL with lens")

    # Transform the initial phase map to the lensless setting
    neural_holography_phase_map = transform_to_neural_holography_setting(
        holoeye_phase_map, prop_dist, wavelength, slm_shape, pixel_pitch
    )
    unpacked_phase_map = neural_holography_phase_map[0, 0, :, :]

    # Simulate the propagation in the lensless setting and show the results
    propped_phase_map = simulated_prop(
        neural_holography_phase_map, neural_holography_asm, prop_dist, wavelength, pixel_pitch,
    )
    show_plot(unpacked_phase_map, propped_phase_map, "CITL without lens")


if __name__ == "__main__":
    simulated_prop_citl_pred()
