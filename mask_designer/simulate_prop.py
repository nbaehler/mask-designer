import matplotlib.pyplot as plt
import torch
from waveprop.fraunhofer import fraunhofer
from waveprop.fresnel import (
    fresnel_conv,
    fresnel_multi_step,
    fresnel_one_step,
    fresnel_two_step,
    shifted_fresnel,
)
from waveprop.rs import (
    angular_spectrum,
    angular_spectrum_np,
    direct_integration,
    fft_di,
)
from waveprop.spherical import spherical_prop
from waveprop.util import ift2

from mask_designer.utils import normalize_mask
from mask_designer.wrapper import fftshift, prop_asm, propagate_field


def holoeye_fraunhofer(field):
    """
    Simulated propagation with a lens (holoeye setup) between slm and target
    plane using Fraunhofer's equation.

    Parameters
    ----------
    field : torch.Tensor
        The field to be propagated

    Returns
    -------
    torch.Tensor
        The result of the propagation at the target plane
    """
    return fftshift(torch.fft.fftn(field, dim=(-2, -1), norm="ortho"))


default_prop_method = holoeye_fraunhofer


def simulate_prop(field, propagation_method=default_prop_method, *args):
    return propagation_method(field, *args)[0, 0, :, :]


def neural_holography_asm(field, prop_dist, wavelength, pixel_pitch):
    """
    Simulated propagation with a no lens (neural holography setup) between slm
    and target plane using the angular spectrum method.

    Parameters
    ----------
    field : torch.Tensor
        The field to be propagated

    Returns
    -------
    torch.Tensor
        The result of the propagation at the target plane
    """
    return propagate_field(
        field, prop_asm, prop_dist, wavelength, pixel_pitch, "ASM", torch.float32, None,
    )


def waveprop_fraunhofer(field, prop_dist, wavelength, pixel_pitch):
    """
    Fraunhofer propagation using waveprop.

    Parameters
    ----------
    field : torch.Tensor
        The field to be propagated
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
    field = field[0, 0, :, :]

    res, _, _ = fraunhofer(u_in=field.numpy(), wv=wavelength, d1=pixel_pitch[0], dz=prop_dist,)

    return torch.from_numpy(res)[None, None, :, :]


def waveprop_asm(field, prop_dist, wavelength, pixel_pitch, device):
    """
    Angular Spectrum Method propagation using waveprop.

    Parameters
    ----------
    field : torch.Tensor
        The field to be propagated
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
    field = field[0, 0, :, :]

    res, _, _ = angular_spectrum(
        u_in=field,
        wv=wavelength,
        d1=pixel_pitch[0],
        dz=prop_dist,
        device=device,
        # out_shift=1,  # TODO check this parameter
    )

    return torch.rot90(ift2(res, delta_f=1), 2)[None, None, :, :]


def waveprop_asm_np(field, prop_dist, wavelength, pixel_pitch):
    """
    Band-limited Angular Spectrum Method for Numerical Simulation of Free-Space
    Propagation in Far and Near Fields propagation using waveprop.

    Parameters
    ----------
    field : torch.Tensor
        The field to be propagated
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
    field = field[0, 0, :, :]

    res, _, _ = angular_spectrum_np(
        u_in=field.numpy(), wv=wavelength, d1=pixel_pitch[0], dz=prop_dist,
    )

    return torch.from_numpy(ift2(res, delta_f=1))[None, None, :, :]


def waveprop_fft_di(field, prop_dist, wavelength, pixel_pitch):
    """
    _summary_ #TODO add those summaries

    Parameters
    ----------
    field : torch.Tensor
        The field to be propagated
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
    field = field[0, 0, :, :]

    res, _, _ = fft_di(u_in=field.numpy(), wv=wavelength, d1=pixel_pitch[0], dz=prop_dist,)

    return torch.from_numpy(res)[None, None, :, :]


def waveprop_direct_integration(field, prop_dist, wavelength, pixel_pitch):
    """
    _summary_

    Parameters
    ----------
    field : torch.Tensor
        The field to be propagated
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
    field = field[0, 0, :, :]

    res = direct_integration(
        u_in=field.numpy(),
        wv=wavelength,
        d1=pixel_pitch[0],
        dz=prop_dist,
        x=[0],  # TODO wrong
        y=[0],  # TODO wrong
    )

    return torch.from_numpy(res)[None, None, :, :]


def waveprop_fresnel_one_step(field, prop_dist, wavelength, pixel_pitch):
    """
    _summary_

    Parameters
    ----------
    field : torch.Tensor
        The field to be propagated
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
    field = field[0, 0, :, :]

    res, _, _ = fresnel_one_step(  # TODO Too small
        u_in=field.numpy(), wv=wavelength, d1=pixel_pitch[0], dz=prop_dist,
    )

    return torch.from_numpy(ift2(res, delta_f=1))[None, None, :, :]


def waveprop_fresnel_two_step(field, prop_dist, wavelength, pixel_pitch):
    """
    _summary_

    Parameters
    ----------
    field : torch.Tensor
        The field to be propagated
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
    field = field[0, 0, :, :]

    res, _, _ = fresnel_two_step(
        u_in=field.numpy(), wv=wavelength, d1=pixel_pitch[0], d2=pixel_pitch[0], dz=prop_dist,
    )

    return torch.from_numpy(ift2(res, delta_f=1))[None, None, :, :]


def waveprop_fresnel_multi_step(field, prop_dist, wavelength, pixel_pitch):
    """
    _summary_

    Parameters
    ----------
    field : torch.Tensor
        The field to be propagated
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
    field = field[0, 0, :, :]

    res, _, _ = fresnel_multi_step(
        u_in=field.numpy(),
        wv=wavelength,
        delta1=pixel_pitch[0],
        deltan=pixel_pitch[0],
        z=prop_dist,
    )

    return torch.from_numpy(ift2(res, delta_f=1))[None, None, :, :]


def waveprop_fresnel_conv(field, prop_dist, wavelength, pixel_pitch, device):
    """
    _summary_

    Parameters
    ----------
    field : torch.Tensor
        The field to be propagated
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
    field = field[0, 0, :, :]

    res, _, _ = fresnel_conv(
        u_in=field, wv=wavelength, d1=pixel_pitch[0], dz=prop_dist, device=device,
    )

    return ift2(res, delta_f=1)[None, None, :, :]


def waveprop_shifted_fresnel(field, prop_dist, wavelength, pixel_pitch):
    """
    _summary_

    Parameters
    ----------
    field : torch.Tensor
        The field to be propagated
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
    field = field[0, 0, :, :]

    res, _, _ = shifted_fresnel(u_in=field.numpy(), wv=wavelength, d1=pixel_pitch[0], dz=prop_dist,)

    return torch.from_numpy(ift2(res, delta_f=1))[None, None, :, :]


def waveprop_spherical(field, prop_dist, wavelength, pixel_pitch, device):
    """
    _summary_

    Parameters
    ----------
    field : torch.Tensor
        The field to be propagated
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
    field = field[0, 0, :, :]

    res, _, _ = spherical_prop(
        u_in=field, wv=wavelength, d1=pixel_pitch[0], dz=prop_dist, device=device
    )

    return res[None, None, :, :]


def plot_mask(mask):
    # Plot
    _, ax = plt.subplots()
    ax.imshow(mask, cmap="gray")
    plt.show()


def plot_fields(field, propped_field, title):
    """
    Plotting utility function.

    Parameters
    ----------
    field : torch.Tensor
        The field before propagation
    propped_field : torch.Tensor
        The field after propagation
    title : String
        The title of the plot
    """
    fig = plt.figure()
    fig.suptitle(title)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.title.set_text("Phase on SLM")
    ax2.title.set_text("Amplitude on SLM")
    ax3.title.set_text("Phase after propagation to screen")
    ax4.title.set_text("Amplitude after propagation to screen")
    ax1.imshow(normalize_mask(field.angle()), cmap="gray")  # TODO normalize?
    ax2.imshow(normalize_mask(field.abs()), cmap="gray")
    ax3.imshow(normalize_mask(propped_field.angle()), cmap="gray")
    ax4.imshow(normalize_mask(propped_field.abs()), cmap="gray")
    plt.show()
