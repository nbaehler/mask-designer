from slm_designer.neural_holography.augmented_image_loader import ImageLoader
from slm_designer.neural_holography.eval import eval
from slm_designer.neural_holography.modules import DPAC, GS, SGD, PhysicalProp
from slm_designer.neural_holography.propagation_ASM import propagation_ASM
from slm_designer.neural_holography.propagation_model import ModelPropagate
from slm_designer.neural_holography.train_model import train_model
from slm_designer.neural_holography.utils import (
    cond_mkdir,
    crop_image,
    fftshift,
    ifftshift,
    make_kernel_gaussian,
    polar_to_rect,
    propagate_field,
    srgb_lin2gamma,
    str2bool,
)

from slm_designer.transform_fields import lensless_to_lens
from slm_designer.utils import extend_to_complex, quantize_phase_pattern


def run_dpac(target_amp, slm_shape, distance, wavelength, pixel_pitch, device):
    # Run Double Phase Amplitude Coding #TODO does not work
    dpac = DPAC(distance, wavelength, pixel_pitch, device=device)
    angles = dpac(target_amp)
    angles = angles.cpu().detach()

    # Extend the computed angles, aka the phase values, to a complex tensor again
    extended = extend_to_complex(angles)

    # Transform the results to the hardware setting using a lens
    final_phase_dpac = lensless_to_lens(
        extended, distance, wavelength, slm_shape, pixel_pitch
    ).angle()

    # Quantize the the angles, aka phase values, to a bit values
    return quantize_phase_pattern(final_phase_dpac)


def run_gs(
    init_phase,
    target_amp,
    iterations,
    slm_shape,
    distance,
    wavelength,
    pixel_pitch,
    device,
):
    # Run Gerchberg-Saxton
    gs = GS(distance, wavelength, pixel_pitch, iterations, device=device)
    angles = gs(target_amp, init_phase).cpu().detach()

    # Extend the computed angles, aka the phase values, to a complex tensor again
    extended = extend_to_complex(angles)

    # Transform the results to the hardware setting using a lens
    final_phase_gs = lensless_to_lens(
        extended, distance, wavelength, slm_shape, pixel_pitch
    ).angle()

    # Quantize the the angles, aka phase values, to a bit values
    return quantize_phase_pattern(final_phase_gs)


def run_sgd(
    init_phase,
    target_amp,
    iterations,
    slm_shape,
    roi_res,
    distance,
    wavelength,
    pixel_pitch,
    device,
):
    # Run Stochastic Gradient Descent based method
    sgd = SGD(distance, wavelength, pixel_pitch, iterations, roi_res, device=device)
    angles = sgd(target_amp, init_phase).cpu().detach()

    # Extend the computed angles, aka the phase values, to a complex tensor again
    extended = extend_to_complex(angles)

    # Transform the results to the hardware setting using a lens
    final_phase_sgd = lensless_to_lens(
        extended, distance, wavelength, slm_shape, pixel_pitch
    ).angle()

    # Quantize the the angles, aka phase values, to a bit values
    return quantize_phase_pattern(final_phase_sgd)
