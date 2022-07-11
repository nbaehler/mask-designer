"""
Simulated propagation using waveprop of the slm pattern generated using the holoeye software.
"""

from slm_designer.experimental_setup import (
    Params,
    params,
    slm_device,
)
from slm_designer.simulated_prop import simulated_prop

from slm_designer.utils import load_phase_map, show_plot
from slm_designer.propagation import (
    holoeye_fraunhofer,
    neural_holography_asm,
    wave_prop_angular_spectrum,
    wave_prop_angular_spectrum_np,
    wave_prop_fft_di,
    wave_prop_direct_integration,
    wave_prop_fraunhofer,
    wave_prop_fresnel_one_step,
    wave_prop_fresnel_two_step,
    wave_prop_fresnel_multi_step,
    wave_prop_fresnel_conv,
    wave_prop_shifted_fresnel,
    wave_prop_spherical,
)
from slm_designer.transform_phase_maps import lens_to_lensless

from slm_controller.hardware import SLMParam, slm_devices


def simulated_prop_waveprop():
    # Define parameters
    prop_dist = params[Params.PROPAGATION_DISTANCE]
    wavelength = params[Params.WAVELENGTH]
    pixel_pitch = slm_devices[slm_device][SLMParam.PIXEL_PITCH]
    slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]

    # Load slm phase map computed with holoeye software
    holoeye_phase_map = load_phase_map()

    # Make it compliant with the data structure used in the project
    unpacked_phase_map = holoeye_phase_map[0, 0, :, :]  # TODO improve this data structure!

    # Simulate the propagation in the lens setting and show the results
    propped_phase_map = simulated_prop(holoeye_phase_map, holoeye_fraunhofer)
    show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with lens")

    # TODO test those, add all?
    # ==========================================================================
    propped_phase_map = simulated_prop(
        holoeye_phase_map, wave_prop_fraunhofer, prop_dist, wavelength, pixel_pitch,
    )
    show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with Fraunhofer")

    propped_phase_map = simulated_prop(
        holoeye_phase_map, wave_prop_angular_spectrum, prop_dist, wavelength, pixel_pitch,
    )
    show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with Angular Spectrum")

    propped_phase_map = simulated_prop(
        holoeye_phase_map, wave_prop_angular_spectrum_np, prop_dist, wavelength, pixel_pitch,
    )
    show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with Angular Spectrum NP")

    propped_phase_map = simulated_prop(
        holoeye_phase_map, wave_prop_fft_di, prop_dist, wavelength, pixel_pitch,  # TODO not working
    )
    show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with FFT Direct")

    propped_phase_map = simulated_prop(
        holoeye_phase_map, wave_prop_direct_integration, prop_dist, wavelength, pixel_pitch,
    )
    show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with Direct Integration")

    propped_phase_map = simulated_prop(
        holoeye_phase_map, wave_prop_fresnel_one_step, prop_dist, wavelength, pixel_pitch,
    )
    show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with Fresnel One Step")

    propped_phase_map = simulated_prop(
        holoeye_phase_map,
        wave_prop_fresnel_two_step,
        prop_dist,
        wavelength,
        pixel_pitch,  # TODO not working
    )
    show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with Fresnel Two Step")

    propped_phase_map = simulated_prop(
        holoeye_phase_map,
        wave_prop_fresnel_multi_step,
        prop_dist,
        wavelength,
        pixel_pitch,  # TODO not working
    )
    show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with Fresnel Multi Step")

    propped_phase_map = simulated_prop(
        holoeye_phase_map, wave_prop_fresnel_conv, prop_dist, wavelength, pixel_pitch,
    )
    show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with Fresnel Convolution")

    propped_phase_map = simulated_prop(
        holoeye_phase_map, wave_prop_shifted_fresnel, prop_dist, wavelength, pixel_pitch,
    )
    show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with Shifted Fresnel")

    propped_phase_map = simulated_prop(
        holoeye_phase_map,
        wave_prop_spherical,
        prop_dist,
        wavelength,
        pixel_pitch,  # TODO not working
    )
    show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with Spherical")

    # ==========================================================================

    # Transform the initial phase map to the lensless setting
    neural_holography_phase_map = lens_to_lensless(
        holoeye_phase_map, prop_dist, wavelength, slm_shape, pixel_pitch
    )
    unpacked_phase_map = neural_holography_phase_map[0, 0, :, :]

    # Simulate the propagation in the lensless setting and show the results
    propped_phase_map = simulated_prop(
        neural_holography_phase_map, neural_holography_asm, prop_dist, wavelength, pixel_pitch,
    )
    show_plot(unpacked_phase_map, propped_phase_map, "Holoeye without lens")


if __name__ == "__main__":
    simulated_prop_waveprop()
