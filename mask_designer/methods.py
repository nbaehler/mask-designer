# TODO change documentation, wrapper and methods in one file introduced too many
# circular imports!!!

from mask_designer.wrapper import DPAC, GS, SGD
from mask_designer.transform_fields import transform_from_neural_holography_setting
from mask_designer.utils import quantize_phase_mask, extend_to_field


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
        The quantized resulting phase mask in 0-255
    """
    # Run Double Phase Amplitude Coding #TODO DPAC does not work
    dpac = DPAC(prop_distance, wavelength, pixel_pitch, device=device)
    angles = dpac(target_amp)
    angles = angles.cpu().detach()

    # Extend the computed angles, aka the phase values, to be a field which is a complex tensor again
    extended = extend_to_field(angles)

    # Transform the results to the hardware setting using a lens
    final_phase_dpac = transform_from_neural_holography_setting(
        extended, prop_distance, wavelength, slm_shape, pixel_pitch
    ).angle()

    # Quantize the fields angles, aka phase values, to a bit values
    return quantize_phase_mask(final_phase_dpac)


def run_gs(
    init_phase, target_amp, iterations, slm_shape, prop_distance, wavelength, pixel_pitch, device,
):
    """
    Run the GS algorithm and quantize the result.

    Parameters
    ----------
    init_phase : torch.Tensor
        The initial random phase mask that is going to be optimized
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
        The quantized resulting phase mask in 0-255
    """
    # Run Gerchberg-Saxton
    gs = GS(prop_distance, wavelength, pixel_pitch, iterations, device=device)
    angles = gs(target_amp, init_phase).cpu().detach()

    # Extend the computed angles, aka the phase values, to be a field which is a complex tensor again
    extended = extend_to_field(angles)

    # Transform the results to the hardware setting using a lens
    final_phase_gs = transform_from_neural_holography_setting(
        extended, prop_distance, wavelength, slm_shape, pixel_pitch
    ).angle()

    # Quantize the fields angles, aka phase values, to a bit values
    return quantize_phase_mask(final_phase_gs)


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
        The initial random phase mask that is going to be optimized
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
        The quantized resulting phase mask in 0-255
    """
    # Run Stochastic Gradient Descent based method
    sgd = SGD(prop_distance, wavelength, pixel_pitch, iterations, roi, device=device)
    angles = sgd(target_amp, init_phase).cpu().detach()

    # Extend the computed angles, aka the phase values, to be a field which is a complex tensor again
    extended = extend_to_field(angles)

    # Transform the results to the hardware setting using a lens
    final_phase_sgd = transform_from_neural_holography_setting(
        extended, prop_distance, wavelength, slm_shape, pixel_pitch
    ).angle()

    # Quantize the fields angles, aka phase values, to a bit values
    return quantize_phase_mask(final_phase_sgd)
