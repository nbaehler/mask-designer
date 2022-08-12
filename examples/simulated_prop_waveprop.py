"""
Simulated propagation using waveprop of the slm pattern generated using the holoeye software.
"""

import torch
from mask_designer.experimental_setup import (
    Params,
    params,
    slm_device,
)
from mask_designer.simulated_prop import simulated_prop

from mask_designer.utils import load_phase_map, show_plot
from mask_designer.propagation import (
    holoeye_fraunhofer,
    neural_holography_asm,
    waveprop_angular_spectrum,
    waveprop_angular_spectrum_np,
    waveprop_fft_di,
    waveprop_direct_integration,
    waveprop_fraunhofer,
    waveprop_fresnel_one_step,
    waveprop_fresnel_two_step,
    waveprop_fresnel_multi_step,
    waveprop_fresnel_conv,
    waveprop_shifted_fresnel,
    waveprop_spherical,
)
from mask_designer.transform_phase_maps import transform_to_neural_holography_setting

from slm_controller.hardware import SLMParam, slm_devices


def simulated_prop_waveprop():
    # Define parameters
    prop_dist = params[Params.PROPAGATION_DISTANCE]
    wavelength = params[Params.WAVELENGTH]
    pixel_pitch = slm_devices[slm_device][SLMParam.PIXEL_PITCH]
    slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]

    # Use GPU if detected in system
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load slm phase map computed with holoeye software
    holoeye_phase_map = load_phase_map()

    # Make it compliant with the data structure used in the project
    unpacked_phase_map = holoeye_phase_map[
        0, 0, :, :
    ]  # TODO improve this data structure, for now only for one image and one channel!

    # Simulate the propagation in the lens setting and show the results
    propped_phase_map = simulated_prop(holoeye_phase_map, holoeye_fraunhofer)
    show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with lens")

    # TODO test those, add all?
    # ==========================================================================
    propped_phase_map = simulated_prop(
        holoeye_phase_map, waveprop_fraunhofer, prop_dist, wavelength, pixel_pitch,
    )
    show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with Fraunhofer")

    propped_phase_map = (
        simulated_prop(
            holoeye_phase_map,
            waveprop_angular_spectrum,
            prop_dist,
            wavelength,
            pixel_pitch,
            device,
        )
        .cpu()
        .detach()
    )
    show_plot(
        unpacked_phase_map, propped_phase_map, "Holoeye with Angular Spectrum",
    )

    propped_phase_map = simulated_prop(
        holoeye_phase_map, waveprop_angular_spectrum_np, prop_dist, wavelength, pixel_pitch,
    )
    show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with Angular Spectrum NP")

    # propped_phase_map = simulated_prop(
    #     holoeye_phase_map, waveprop_fft_di, prop_dist, wavelength, pixel_pitch,  # TODO not working
    # )
    # show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with FFT Direct")

    propped_phase_map = simulated_prop(
        holoeye_phase_map, waveprop_direct_integration, prop_dist, wavelength, pixel_pitch,
    )
    show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with Direct Integration")

    propped_phase_map = simulated_prop(
        holoeye_phase_map,
        waveprop_fresnel_one_step,  # TODO this one seems to be the only one that captures scale as well???
        prop_dist,
        wavelength,
        pixel_pitch,
    )
    show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with Fresnel One Step")

    # propped_phase_map = simulated_prop(
    #     holoeye_phase_map,
    #     waveprop_fresnel_two_step,
    #     prop_dist,
    #     wavelength,
    #     pixel_pitch,  # TODO not working
    # )
    # show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with Fresnel Two Step")

    # propped_phase_map = simulated_prop(
    #     holoeye_phase_map,
    #     waveprop_fresnel_multi_step,
    #     prop_dist,
    #     wavelength,
    #     pixel_pitch,  # TODO not working
    # )
    # show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with Fresnel Multi Step")

    propped_phase_map = (
        simulated_prop(
            holoeye_phase_map, waveprop_fresnel_conv, prop_dist, wavelength, pixel_pitch, device,
        )
        .cpu()
        .detach()
    )
    show_plot(
        unpacked_phase_map, propped_phase_map, "Holoeye with Fresnel Convolution",
    )

    propped_phase_map = simulated_prop(
        holoeye_phase_map, waveprop_shifted_fresnel, prop_dist, wavelength, pixel_pitch,
    )
    show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with Shifted Fresnel")

    # propped_phase_map = (
    #     simulated_prop(
    #         holoeye_phase_map,
    #         waveprop_spherical,
    #         prop_dist,
    #         wavelength,
    #         pixel_pitch,  # TODO not working
    #         device,
    #     )
    #     .cpu()
    #     .detach()
    # )
    # show_plot(unpacked_phase_map, propped_phase_map, "Holoeye with Spherical")

    # ==========================================================================

    # Transform the initial phase map to the lensless setting
    neural_holography_phase_map = transform_to_neural_holography_setting(
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
