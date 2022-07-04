"""
Simulated propagation using waveprop of the slm pattern generated using the holoeye software.
TODO contains a lot waveprop methods than still need some testing/bug fixing
"""

from slm_designer.experimental_setup import (
    PhysicalParams,
    physical_params,
    slm_device,
)

from slm_designer.utils import load_holoeye_slm_pattern, show_plot
from slm_designer.simulated_prop import (
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
from slm_designer.transform_fields import lens_to_lensless

from slm_controller.hardware import (
    SLMParam,
    slm_devices,
)


def simulated_prop_waveprop():
    # Define parameters
    prop_dist = physical_params[PhysicalParams.PROPAGATION_DISTANCE]
    wavelength = physical_params[PhysicalParams.WAVELENGTH]
    slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]
    pixel_pitch = slm_devices[slm_device][SLMParam.PIXEL_PITCH]

    # Load slm phase map computed with holoeye software
    holoeye_slm_field = load_holoeye_slm_pattern()

    # Make it compliant with the data structure used in the project
    slm_field = holoeye_slm_field[0, 0, :, :]  # TODO improve this data structure!

    # Simulate the propagation in the lens setting and show the results
    propped_slm_field = holoeye_fraunhofer(holoeye_slm_field)[0, 0, :, :]
    show_plot(slm_field, propped_slm_field, "Holoeye with lens")

    # TODO test those
    # ==========================================================================
    propped_slm_field = wave_prop_fraunhofer(
        holoeye_slm_field, prop_dist, wavelength, pixel_pitch,
    )[0, 0, :, :]
    show_plot(slm_field, propped_slm_field, "Holoeye with Fraunhofer")

    propped_slm_field = wave_prop_angular_spectrum(
        holoeye_slm_field, prop_dist, wavelength, pixel_pitch,
    )[0, 0, :, :]
    show_plot(slm_field, propped_slm_field, "Holoeye with Angular Spectrum")

    propped_slm_field = wave_prop_angular_spectrum_np(
        holoeye_slm_field, prop_dist, wavelength, pixel_pitch,
    )[0, 0, :, :]
    show_plot(slm_field, propped_slm_field, "Holoeye with Angular Spectrum NP")

    propped_slm_field = wave_prop_fft_di(
        holoeye_slm_field, prop_dist, wavelength, pixel_pitch,  # TODO not working
    )[
        0, 0, :, :
    ]  # TODO add those! All?
    show_plot(slm_field, propped_slm_field, "Holoeye with FFT Direct")

    propped_slm_field = wave_prop_direct_integration(
        holoeye_slm_field, prop_dist, wavelength, pixel_pitch,
    )[0, 0, :, :]
    show_plot(slm_field, propped_slm_field, "Holoeye with Direct Integration")

    propped_slm_field = wave_prop_fresnel_one_step(
        holoeye_slm_field, prop_dist, wavelength, pixel_pitch,
    )[0, 0, :, :]
    show_plot(slm_field, propped_slm_field, "Holoeye with Fresnel One Step")

    propped_slm_field = wave_prop_fresnel_two_step(
        holoeye_slm_field, prop_dist, wavelength, pixel_pitch,  # TODO not working
    )[0, 0, :, :]
    show_plot(slm_field, propped_slm_field, "Holoeye with Fresnel Two Step")

    propped_slm_field = wave_prop_fresnel_multi_step(
        holoeye_slm_field, prop_dist, wavelength, pixel_pitch,  # TODO not working
    )[0, 0, :, :]
    show_plot(slm_field, propped_slm_field, "Holoeye with Fresnel Multi Step")

    propped_slm_field = wave_prop_fresnel_conv(
        holoeye_slm_field, prop_dist, wavelength, pixel_pitch,
    )[0, 0, :, :]
    show_plot(slm_field, propped_slm_field, "Holoeye with Fresnel Convolution")

    propped_slm_field = wave_prop_shifted_fresnel(
        holoeye_slm_field, prop_dist, wavelength, pixel_pitch,
    )[0, 0, :, :]
    show_plot(slm_field, propped_slm_field, "Holoeye with Shifted Fresnel")

    propped_slm_field = wave_prop_spherical(
        holoeye_slm_field, prop_dist, wavelength, pixel_pitch,  # TODO not working
    )[0, 0, :, :]
    show_plot(slm_field, propped_slm_field, "Holoeye with Spherical")

    # ==========================================================================

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
    simulated_prop_waveprop()
