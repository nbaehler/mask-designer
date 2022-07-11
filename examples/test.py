from slm_designer.experimental_setup import (
    Params,
    params,
    slm_device,
)

from slm_designer.utils import load_phase_map, show_plot
from slm_designer.simulated_prop import simulated_prop, plot_sim_result

from slm_designer.propagation import neural_holography_asm
from slm_designer.transform_phase_maps import transform_to_neural_holography_setting

from slm_controller.hardware import (
    SLMParam,
    slm_devices,
)


def test():
    # Define parameters
    prop_dist = params[Params.PROPAGATION_DISTANCE]
    wavelength = params[Params.WAVELENGTH]
    pixel_pitch = slm_devices[slm_device][SLMParam.PIXEL_PITCH]
    slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]

    # Load slm phase map computed with holoeye software
    holoeye_phase_map = load_phase_map()

    # Make it compliant with the data structure used in the project
    phase_map = holoeye_phase_map[0, 0, :, :]

    # Simulate the propagation in the lens setting and show the results
    propped_phase_map = simulated_prop(holoeye_phase_map)
    show_plot(phase_map, propped_phase_map, "Holoeye with lens")

    plot_sim_result(simulated_prop(holoeye_phase_map))

    # Transform the initial phase map to the lensless setting
    holoeye_phase_map = transform_to_neural_holography_setting(
        holoeye_phase_map, prop_dist, wavelength, slm_shape, pixel_pitch
    )
    phase_map = holoeye_phase_map[0, 0, :, :]

    # Simulate the propagation in the lensless setting and show the results
    propped_phase_map = simulated_prop(
        holoeye_phase_map, neural_holography_asm, prop_dist, wavelength, pixel_pitch,
    )
    show_plot(phase_map, propped_phase_map, "Holoeye without lens")


if __name__ == "__main__":
    test()
