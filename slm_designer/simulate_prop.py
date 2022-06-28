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


def lens_prop(slm_field):
    """
    Simulated propagation with a lens (holoeye setting) between slm and target plane.

    Parameters
    ----------
    slm_field : torch.Tensor
        The phase map to be propagated

    Returns
    -------
    torch.Tensor
        The result of the propagation at the target plane
    """
    # return fftshift(slm_field) # TODO check this
    return fftshift(
        torch.fft.fftn(slm_field, dim=(-2, -1), norm="ortho")
    )  # TODO no wavelength etc???


def lensless_prop(slm_field, prop_dist, wavelength, slm_pitch):
    """
    Simulated propagation with a no lens (neural holography setting) between slm and target plane.

    Parameters
    ----------
    slm_field : torch.Tensor
        The phase map to be propagated

    Returns
    -------
    torch.Tensor
        The result of the propagation at the target plane
    """
    return propagate_field(
        slm_field,
        propagation_ASM,
        prop_dist,
        wavelength,
        slm_pitch,
        "ASM",
        torch.float32,
        None,
    )


def wave_prop_fraunhofer(slm_field, prop_dist, wavelength, slm_pitch):
    slm_field = slm_field[0, 0, :, :]

    res, _, _ = fraunhofer(
        u_in=slm_field.numpy(), wv=wavelength, d1=slm_pitch[0], dz=prop_dist,
    )

    return torch.from_numpy(res)[None, None, :, :]


def wave_prop_angular_spectrum(slm_field, prop_dist, wavelength, slm_pitch):
    slm_field = slm_field[0, 0, :, :]

    res, _, _ = angular_spectrum(  # TODO flipped
        u_in=slm_field.numpy(),
        wv=wavelength,
        d1=slm_pitch[0],
        dz=prop_dist,
        # out_shift=1,  # TODO check this
    )

    return torch.from_numpy(ift2(res, delta_f=1))[None, None, :, :]
    # return torch.from_numpy(res)[None, None, :, :]


def wave_prop_angular_spectrum_np(slm_field, prop_dist, wavelength, slm_pitch):
    slm_field = slm_field[0, 0, :, :]

    res, _, _ = angular_spectrum_np(
        u_in=slm_field.numpy(), wv=wavelength, d1=slm_pitch[0], dz=prop_dist,
    )

    return torch.from_numpy(ift2(res, delta_f=1))[None, None, :, :]


def wave_prop_fft_di(slm_field, prop_dist, wavelength, slm_pitch):
    slm_field = slm_field[0, 0, :, :]

    res, _, _ = fft_di(
        u_in=slm_field.numpy(), wv=wavelength, d1=slm_pitch[0], dz=prop_dist,
    )

    return torch.from_numpy(res)[None, None, :, :]


def wave_prop_direct_integration(slm_field, prop_dist, wavelength, slm_pitch):
    slm_field = slm_field[0, 0, :, :]

    res = direct_integration(
        u_in=slm_field.numpy(),
        wv=wavelength,
        d1=slm_pitch[0],
        dz=prop_dist,
        x=[0],  # TODO wrong
        y=[0],  # TODO wrong
    )

    return torch.from_numpy(res)[None, None, :, :]


def wave_prop_fresnel_one_step(slm_field, prop_dist, wavelength, slm_pitch):
    slm_field = slm_field[0, 0, :, :]

    res, _, _ = fresnel_one_step(  # TODO Too small
        u_in=slm_field.numpy(), wv=wavelength, d1=slm_pitch[0], dz=prop_dist,
    )

    return torch.from_numpy(ift2(res, delta_f=1))[None, None, :, :]


def wave_prop_fresnel_two_step(slm_field, prop_dist, wavelength, slm_pitch):
    slm_field = slm_field[0, 0, :, :]

    res, _, _ = fresnel_two_step(
        u_in=slm_field.numpy(),
        wv=wavelength,
        d1=slm_pitch[0],
        d2=slm_pitch[0],
        dz=prop_dist,
    )

    return torch.from_numpy(ift2(res, delta_f=1))[None, None, :, :]


def wave_prop_fresnel_multi_step(slm_field, prop_dist, wavelength, slm_pitch):
    slm_field = slm_field[0, 0, :, :]

    res, _, _ = fresnel_multi_step(
        u_in=slm_field.numpy(),
        wv=wavelength,
        delta1=slm_pitch[0],
        deltan=slm_pitch[0],
        z=prop_dist,
    )

    return torch.from_numpy(ift2(res, delta_f=1))[None, None, :, :]


def wave_prop_fresnel_conv(slm_field, prop_dist, wavelength, slm_pitch):
    slm_field = slm_field[0, 0, :, :]

    res, _, _ = fresnel_conv(
        u_in=slm_field.numpy(), wv=wavelength, d1=slm_pitch[0], dz=prop_dist,
    )

    return torch.from_numpy(ift2(res, delta_f=1))[None, None, :, :]


def wave_prop_shifted_fresnel(slm_field, prop_dist, wavelength, slm_pitch):
    slm_field = slm_field[0, 0, :, :]

    res, _, _ = shifted_fresnel(
        u_in=slm_field.numpy(), wv=wavelength, d1=slm_pitch[0], dz=prop_dist,
    )

    return torch.from_numpy(ift2(res, delta_f=1))[None, None, :, :]


def wave_prop_spherical(slm_field, prop_dist, wavelength, slm_pitch):
    slm_field = slm_field[0, 0, :, :]

    res, _, _ = spherical_prop(
        u_in=slm_field.numpy(), wv=wavelength, d1=slm_pitch[0], dz=prop_dist,
    )

    return torch.from_numpy(res)[None, None, :, :]
