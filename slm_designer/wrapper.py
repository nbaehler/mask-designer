from slm_designer.neural_holography.augmented_image_loader import (
    ImageLoader,
    get_image_filenames,
)
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

from slm_designer.transform_phase_maps import (
    transform_from_neural_holography_setting,
)  # TODO circular dependency but it works
from slm_designer.utils import extend_to_complex, quantize_phase_pattern


def run_dpac(target_amp, slm_shape, prop_distance, wavelength, pixel_pitch, device):
    """
    Run the DPAC algorithm and quantize the result.

    Parameters
    ----------
    target_amp : torch.Tensor
        The target amplitude
    slm_shape : tuple(int)
        The shape or the resolution of the SLM
    prop_dist : float
        The propagation distance from the SLM to the target plane
    wavelength : float
        The wavelength of the light
    pixel_pitch : float
        The pixel pitch of the SLM
    device : String
        The device that is used, either cpu or cuda

    Returns
    -------
    numpy.ndarray
        The quantized resulting phase map in 0-255
    """
    # Run Double Phase Amplitude Coding #TODO DPAC does not work
    dpac = DPAC(prop_distance, wavelength, pixel_pitch, device=device)
    angles = dpac(target_amp)
    angles = angles.cpu().detach()

    # Extend the computed angles, aka the phase values, to a complex tensor again
    extended = extend_to_complex(angles)

    # Transform the results to the hardware setting using a lens
    final_phase_dpac = transform_from_neural_holography_setting(
        extended, prop_distance, wavelength, slm_shape, pixel_pitch
    ).angle()

    # Quantize the the angles, aka phase values, to a bit values
    return quantize_phase_pattern(final_phase_dpac)


def run_gs(
    init_phase,
    target_amp,
    iterations,
    slm_shape,
    prop_distance,
    wavelength,
    pixel_pitch,
    device,
):
    """
    Run the GS algorithm and quantize the result.

    Parameters
    ----------
    init_phase : torch.Tensor
        The initial random phase map that is going to be optimized
    target_amp : torch.Tensor
        The target amplitude
    iterations : int
        The number of iterations to run
    slm_shape : tuple(int)
        The shape or the resolution of the SLM
    prop_dist : float
        The propagation distance from the SLM to the target plane
    wavelength : float
        The wavelength of the light
    pixel_pitch : float
        The pixel pitch of the SLM
    device : String
        The device that is used, either cpu or cuda

    Returns
    -------
    numpy.ndarray
        The quantized resulting phase map in 0-255
    """
    # Run Gerchberg-Saxton
    gs = GS(prop_distance, wavelength, pixel_pitch, iterations, device=device)
    angles = gs(target_amp, init_phase).cpu().detach()

    # Extend the computed angles, aka the phase values, to a complex tensor again
    extended = extend_to_complex(angles)

    # Transform the results to the hardware setting using a lens
    final_phase_gs = transform_from_neural_holography_setting(
        extended, prop_distance, wavelength, slm_shape, pixel_pitch
    ).angle()

    # Quantize the the angles, aka phase values, to a bit values
    return quantize_phase_pattern(final_phase_gs)


def run_sgd(
    init_phase,
    target_amp,
    iterations,
    slm_shape,
    roi,
    prop_distance,
    wavelength,
    pixel_pitch,
    device,
):
    """
    Run the SGD algorithm and quantize the result.

    Parameters
    ----------
    init_phase : torch.Tensor
        The initial random phase map that is going to be optimized
    target_amp : torch.Tensor
        The target amplitude
    iterations : int
        The number of iterations to run
    slm_shape : tuple(int)
        The shape or the resolution of the SLM
    roi : tuple(int)
        The region of interest in which errors are more strongly penalized
    prop_dist : float
        The propagation distance from the SLM to the target plane
    wavelength : float
        The wavelength of the light
    pixel_pitch : float
        The pixel pitch of the SLM
    device : String
        The device that is used, either cpu or cuda

    Returns
    -------
    numpy.ndarray
        The quantized resulting phase map in 0-255
    """
    # Run Stochastic Gradient Descent based method
    sgd = SGD(prop_distance, wavelength, pixel_pitch, iterations, roi, device=device)
    angles = sgd(target_amp, init_phase).cpu().detach()

    # Extend the computed angles, aka the phase values, to a complex tensor again
    extended = extend_to_complex(angles)

    # Transform the results to the hardware setting using a lens
    final_phase_sgd = transform_from_neural_holography_setting(
        extended, prop_distance, wavelength, slm_shape, pixel_pitch
    ).angle()

    # Quantize the the angles, aka phase values, to a bit values
    return quantize_phase_pattern(final_phase_sgd)
