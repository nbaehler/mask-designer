"""
Simulated propagation using waveprop of the phase mask generated using the holoeye software.
"""

import sys
from os.path import abspath, dirname, join

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)

import torch
from mask_designer.experimental_setup import Params, params, slm_device
from mask_designer.simulate_prop import (
    holoeye_fraunhofer,
    neural_holography_asm,
    plot_fields,
    simulate_prop,
    waveprop_asm,
    waveprop_asm_np,
    waveprop_direct_integration,
    waveprop_fft_di,
    waveprop_fraunhofer,
    waveprop_fresnel_conv,
    waveprop_fresnel_multi_step,
    waveprop_fresnel_one_step,
    waveprop_fresnel_two_step,
    waveprop_shifted_fresnel,
    waveprop_spherical,
)
from mask_designer.transform_fields import transform_to_neural_holography_setting
from mask_designer.utils import load_field
from slm_controller.hardware import SLMParam, slm_devices


def main():
    # Define parameters
    prop_dist = params[Params.PROPAGATION_DISTANCE]
    wavelength = params[Params.WAVELENGTH]
    pixel_pitch = slm_devices[slm_device][SLMParam.PIXEL_PITCH]
    slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]

    # Use GPU if detected in system
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the field computed with holoeye software
    holoeye_field = load_field()

    # Make it compliant with the data structure used in the project
    unpacked_field = holoeye_field[
        0, 0, :, :
    ]  # TODO improve this data structure, for now only for one image and one channel!

    # Simulate the propagation in the lens setting and show the results
    propped_field = simulate_prop(holoeye_field, holoeye_fraunhofer)
    plot_fields(unpacked_field, propped_field, "Holoeye with lens")

    # TODO test those, add all?
    # ==========================================================================
    propped_field = simulate_prop(
        holoeye_field, waveprop_fraunhofer, prop_dist, wavelength, pixel_pitch,
    )
    plot_fields(unpacked_field, propped_field, "Holoeye with Fraunhofer")

    propped_field = (
        simulate_prop(holoeye_field, waveprop_asm, prop_dist, wavelength, pixel_pitch, device,)
        .cpu()
        .detach()
    )
    plot_fields(
        unpacked_field, propped_field, "Holoeye with Angular Spectrum",
    )

    propped_field = simulate_prop(
        holoeye_field, waveprop_asm_np, prop_dist, wavelength, pixel_pitch,
    )
    plot_fields(unpacked_field, propped_field, "Holoeye with Angular Spectrum NP")

    propped_field = simulate_prop(
        holoeye_field, waveprop_fft_di, prop_dist, wavelength, pixel_pitch,  # TODO not working
    )
    plot_fields(unpacked_field, propped_field, "Holoeye with FFT Direct")

    propped_field = simulate_prop(
        holoeye_field, waveprop_direct_integration, prop_dist, wavelength, pixel_pitch,
    )
    plot_fields(unpacked_field, propped_field, "Holoeye with Direct Integration")

    propped_field = simulate_prop(
        holoeye_field,
        waveprop_fresnel_one_step,  # TODO this one seems to be the only one that captures scale
        prop_dist,
        wavelength,
        pixel_pitch,
    )
    plot_fields(unpacked_field, propped_field, "Holoeye with Fresnel One Step")

    propped_field = simulate_prop(
        holoeye_field,
        waveprop_fresnel_two_step,
        prop_dist,
        wavelength,
        pixel_pitch,  # TODO not working
    )
    plot_fields(unpacked_field, propped_field, "Holoeye with Fresnel Two Step")

    propped_field = simulate_prop(
        holoeye_field,
        waveprop_fresnel_multi_step,
        prop_dist,
        wavelength,
        pixel_pitch,  # TODO not working
    )
    plot_fields(unpacked_field, propped_field, "Holoeye with Fresnel Multi Step")

    propped_field = (
        simulate_prop(
            holoeye_field, waveprop_fresnel_conv, prop_dist, wavelength, pixel_pitch, device,
        )
        .cpu()
        .detach()
    )
    plot_fields(
        unpacked_field, propped_field, "Holoeye with Fresnel Convolution",
    )

    propped_field = simulate_prop(
        holoeye_field, waveprop_shifted_fresnel, prop_dist, wavelength, pixel_pitch,
    )
    plot_fields(unpacked_field, propped_field, "Holoeye with Shifted Fresnel")

    propped_field = (
        simulate_prop(
            holoeye_field,
            waveprop_spherical,
            prop_dist,
            wavelength,
            pixel_pitch,  # TODO not working
            device,
        )
        .cpu()
        .detach()
    )
    plot_fields(unpacked_field, propped_field, "Holoeye with Spherical")

    # ==========================================================================

    # Transform the initial phase mask to the lensless setting
    neural_holography_field = transform_to_neural_holography_setting(
        holoeye_field, prop_dist, wavelength, slm_shape, pixel_pitch
    )
    unpacked_field = neural_holography_field[0, 0, :, :]

    # Simulate the propagation in the lensless setting and show the results
    propped_field = simulate_prop(
        neural_holography_field, neural_holography_asm, prop_dist, wavelength, pixel_pitch,
    )
    plot_fields(unpacked_field, propped_field, "Holoeye without lens")


if __name__ == "__main__":
    main()
