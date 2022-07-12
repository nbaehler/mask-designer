import torch

from slm_designer.wrapper import fftshift, propagate_field, propagation_ASM
from waveprop.fraunhofer import fraunhofer
from waveprop.fresnel import (
    fresnel_conv,
    fresnel_one_step,
    fresnel_two_step,
    fresnel_multi_step,
    shifted_fresnel,
)
from waveprop.rs import (
    angular_spectrum,
    direct_integration,
    fft_di,
    angular_spectrum_np,
)
from waveprop.spherical import spherical_prop
from waveprop.util import ift2


def holoeye_fraunhofer(phase_map):
    """
    Simulated propagation with a lens (holoeye setup) between slm and target
    plane using Fraunhofer's equation.

    Parameters
    ----------
    phase_map : torch.Tensor
        The phase map to be propagated

    Returns
    -------
    torch.Tensor
        The result of the propagation at the target plane
    """
    return fftshift(torch.fft.fftn(phase_map, dim=(-2, -1), norm="ortho"))


def neural_holography_asm(phase_map, prop_dist, wavelength, pixel_pitch):
    """
    Simulated propagation with a no lens (neural holography setup) between slm
    and target plane using the angular spectrum method.

    Parameters
    ----------
    phase_map : torch.Tensor
        The phase map to be propagated

    Returns
    -------
    torch.Tensor
        The result of the propagation at the target plane
    """
    return propagate_field(
        phase_map,
        propagation_ASM,
        prop_dist,
        wavelength,
        pixel_pitch,
        "ASM",
        torch.float32,
        None,
    )


def waveprop_fraunhofer(phase_map, prop_dist, wavelength, pixel_pitch):
    """
    Fraunhofer propagation using waveprop.

    Parameters
    ----------
    phase_map : torch.Tensor
        The phase map to be propagated
    prop_dist : float
        The propagation distance from the SLM to the target plane
    wavelength : float
        The wavelength of the light
    pixel_pitch : float
        The pixel pitch of the SLM

    Returns
    -------
    torch.Tensor
        The result of the propagation at the target plane
    """
    phase_map = phase_map[0, 0, :, :]

    res, _, _ = fraunhofer(
        u_in=phase_map.numpy(), wv=wavelength, d1=pixel_pitch[0], dz=prop_dist,
    )

    return torch.from_numpy(res)[None, None, :, :]


def propagator_waveprop_angular_spectrum(  # TODO still buggy
    u_in,
    feature_size,
    wavelength,
    z,
    # linear_conv=True,
    # padtype="zero",
    return_H=False,
    precomped_H=None,
    return_H_exp=False,
    precomped_H_exp=None,
    dtype=torch.float32,
):
    if return_H or return_H_exp:
        return angular_spectrum(
            u_in=u_in[0, 0, :, :],
            wv=wavelength,
            d1=feature_size[0],
            dz=z,
            # linear_conv=True, # TODO check those two parameters
            # padtype="zero",
            return_H=return_H,
            H=precomped_H,
            return_H_exp=return_H_exp,
            H_exp=precomped_H_exp,
            dtype=dtype,
            device="cuda",  # TODO always on gpu?
        )

    res, _, _ = angular_spectrum(
        u_in=u_in[0, 0, :, :],
        wv=wavelength,
        d1=feature_size[0],
        dz=z,
        # linear_conv=True, # TODO check those two parameters
        # padtype="zero",
        return_H=False,
        H=precomped_H,
        return_H_exp=False,
        H_exp=precomped_H_exp,
        dtype=dtype,
        device="cuda",  # TODO always on gpu?
    )

    return torch.rot90(ift2(res, delta_f=1), 2)[
        None, None, :, :
    ]  # TODO Temporary fix for flipped


def waveprop_angular_spectrum(phase_map, prop_dist, wavelength, pixel_pitch, device):
    """
    Angular Spectrum Method propagation using waveprop.

    Parameters
    ----------
    phase_map : torch.Tensor
        The phase map to be propagated
    prop_dist : float
        The propagation distance from the SLM to the target plane
    wavelength : float
        The wavelength of the light
    pixel_pitch : float
        The pixel pitch of the SLM

    Returns
    -------
    torch.Tensor
        The result of the propagation at the target plane
    """
    phase_map = phase_map[0, 0, :, :]

    res, _, _ = angular_spectrum(
        u_in=phase_map,
        wv=wavelength,
        d1=pixel_pitch[0],
        dz=prop_dist,
        device=device,
        # out_shift=1,  # TODO check this parameter
    )

    # return fftshift(res[None, None, :, :])

    return torch.rot90(ift2(res, delta_f=1), 2)[
        None, None, :, :
    ]  # TODO Temporary fix for flipped, copy fixes negative stride issue


def waveprop_angular_spectrum_np(phase_map, prop_dist, wavelength, pixel_pitch):
    """
    Band-limited Angular Spectrum Method for Numerical Simulation of Free-Space
    Propagation in Far and Near Fields propagation using waveprop.

    Parameters
    ----------
    phase_map : torch.Tensor
        The phase map to be propagated
    prop_dist : float
        The propagation distance from the SLM to the target plane
    wavelength : float
        The wavelength of the light
    pixel_pitch : float
        The pixel pitch of the SLM

    Returns
    -------
    torch.Tensor
        The result of the propagation at the target plane
    """
    phase_map = phase_map[0, 0, :, :]

    res, _, _ = angular_spectrum_np(
        u_in=phase_map.numpy(), wv=wavelength, d1=pixel_pitch[0], dz=prop_dist,
    )

    return torch.from_numpy(ift2(res, delta_f=1))[None, None, :, :]


def waveprop_fft_di(phase_map, prop_dist, wavelength, pixel_pitch):
    """
    _summary_ #TODO add those summaries

    Parameters
    ----------
    phase_map : torch.Tensor
        The phase map to be propagated
    prop_dist : float
        The propagation distance from the SLM to the target plane
    wavelength : float
        The wavelength of the light
    pixel_pitch : float
        The pixel pitch of the SLM

    Returns
    -------
    torch.Tensor
        The result of the propagation at the target plane
    """
    phase_map = phase_map[0, 0, :, :]

    res, _, _ = fft_di(
        u_in=phase_map.numpy(), wv=wavelength, d1=pixel_pitch[0], dz=prop_dist,
    )

    return torch.from_numpy(res)[None, None, :, :]


def waveprop_direct_integration(phase_map, prop_dist, wavelength, pixel_pitch):
    """
    _summary_

    Parameters
    ----------
    phase_map : torch.Tensor
        The phase map to be propagated
    prop_dist : float
        The propagation distance from the SLM to the target plane
    wavelength : float
        The wavelength of the light
    pixel_pitch : float
        The pixel pitch of the SLM

    Returns
    -------
    torch.Tensor
        The result of the propagation at the target plane
    """
    phase_map = phase_map[0, 0, :, :]

    res = direct_integration(
        u_in=phase_map.numpy(),
        wv=wavelength,
        d1=pixel_pitch[0],
        dz=prop_dist,
        x=[0],  # TODO wrong
        y=[0],  # TODO wrong
    )

    return torch.from_numpy(res)[None, None, :, :]


def waveprop_fresnel_one_step(phase_map, prop_dist, wavelength, pixel_pitch):
    """
    _summary_

    Parameters
    ----------
    phase_map : torch.Tensor
        The phase map to be propagated
    prop_dist : float
        The propagation distance from the SLM to the target plane
    wavelength : float
        The wavelength of the light
    pixel_pitch : float
        The pixel pitch of the SLM

    Returns
    -------
    torch.Tensor
        The result of the propagation at the target plane
    """
    phase_map = phase_map[0, 0, :, :]

    res, _, _ = fresnel_one_step(  # TODO Too small
        u_in=phase_map.numpy(), wv=wavelength, d1=pixel_pitch[0], dz=prop_dist,
    )

    return torch.from_numpy(ift2(res, delta_f=1))[None, None, :, :]


def waveprop_fresnel_two_step(phase_map, prop_dist, wavelength, pixel_pitch):
    """
    _summary_

    Parameters
    ----------
    phase_map : torch.Tensor
        The phase map to be propagated
    prop_dist : float
        The propagation distance from the SLM to the target plane
    wavelength : float
        The wavelength of the light
    pixel_pitch : float
        The pixel pitch of the SLM

    Returns
    -------
    torch.Tensor
        The result of the propagation at the target plane
    """
    phase_map = phase_map[0, 0, :, :]

    res, _, _ = fresnel_two_step(
        u_in=phase_map.numpy(),
        wv=wavelength,
        d1=pixel_pitch[0],
        d2=pixel_pitch[0],
        dz=prop_dist,
    )

    return torch.from_numpy(ift2(res, delta_f=1))[None, None, :, :]


def waveprop_fresnel_multi_step(phase_map, prop_dist, wavelength, pixel_pitch):
    """
    _summary_

    Parameters
    ----------
    phase_map : torch.Tensor
        The phase map to be propagated
    prop_dist : float
        The propagation distance from the SLM to the target plane
    wavelength : float
        The wavelength of the light
    pixel_pitch : float
        The pixel pitch of the SLM

    Returns
    -------
    torch.Tensor
        The result of the propagation at the target plane
    """
    phase_map = phase_map[0, 0, :, :]

    res, _, _ = fresnel_multi_step(
        u_in=phase_map.numpy(),
        wv=wavelength,
        delta1=pixel_pitch[0],
        deltan=pixel_pitch[0],
        z=prop_dist,
    )

    return torch.from_numpy(ift2(res, delta_f=1))[None, None, :, :]


def waveprop_fresnel_conv(phase_map, prop_dist, wavelength, pixel_pitch, device):
    """
    _summary_

    Parameters
    ----------
    phase_map : torch.Tensor
        The phase map to be propagated
    prop_dist : float
        The propagation distance from the SLM to the target plane
    wavelength : float
        The wavelength of the light
    pixel_pitch : float
        The pixel pitch of the SLM

    Returns
    -------
    torch.Tensor
        The result of the propagation at the target plane
    """
    phase_map = phase_map[0, 0, :, :]

    res, _, _ = fresnel_conv(
        u_in=phase_map, wv=wavelength, d1=pixel_pitch[0], dz=prop_dist, device=device,
    )

    return ift2(res, delta_f=1)[None, None, :, :]


def waveprop_shifted_fresnel(phase_map, prop_dist, wavelength, pixel_pitch):
    """
    _summary_

    Parameters
    ----------
    phase_map : torch.Tensor
        The phase map to be propagated
    prop_dist : float
        The propagation distance from the SLM to the target plane
    wavelength : float
        The wavelength of the light
    pixel_pitch : float
        The pixel pitch of the SLM

    Returns
    -------
    torch.Tensor
        The result of the propagation at the target plane
    """
    phase_map = phase_map[0, 0, :, :]

    res, _, _ = shifted_fresnel(
        u_in=phase_map.numpy(), wv=wavelength, d1=pixel_pitch[0], dz=prop_dist,
    )

    return torch.from_numpy(ift2(res, delta_f=1))[None, None, :, :]


def waveprop_spherical(phase_map, prop_dist, wavelength, pixel_pitch, device):
    """
    _summary_

    Parameters
    ----------
    phase_map : torch.Tensor
        The phase map to be propagated
    prop_dist : float
        The propagation distance from the SLM to the target plane
    wavelength : float
        The wavelength of the light
    pixel_pitch : float
        The pixel pitch of the SLM

    Returns
    -------
    torch.Tensor
        The result of the propagation at the target plane
    """
    phase_map = phase_map[0, 0, :, :]

    res, _, _ = spherical_prop(
        u_in=phase_map, wv=wavelength, d1=pixel_pitch[0], dz=prop_dist, device=device
    )

    return res[None, None, :, :]
