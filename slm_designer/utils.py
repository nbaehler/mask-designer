import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def _cell_slice(_slice, cell_m):
    """
    Convert slice indexing in meters to slice indexing in cells.

    author: Eric Bezzam,
    email: ebezzam@gmail.com,
    GitHub: https://github.com/ebezzam

    Parameters
    ----------
    _slice : slice
        Original slice in meters.
    cell_m : float
        Dimension of cell in meters.
    """
    start = None if _slice.start is None else _m_to_cell_idx(_slice.start, cell_m)
    stop = _m_to_cell_idx(_slice.stop, cell_m) if _slice.stop is not None else None
    step = _m_to_cell_idx(_slice.step, cell_m) if _slice.step is not None else None
    return slice(start, stop, step)


def _m_to_cell_idx(val, cell_m):
    """
    Convert location to cell index.

    author: Eric Bezzam,
    email: ebezzam@gmail.com,
    GitHub: https://github.com/ebezzam

    Parameters
    ----------
    val : float
        Location in meters.
    cell_m : float
        Dimension of cell in meters.
    """
    return int(val / cell_m)


# def si2cell(val: np.ndarray, cell_m): # TODO unused
#     """
#     Convert locations to cell index.
#
#     author: Eric Bezzam,
#     email: ebezzam@gmail.com,
#     GitHub: https://github.com/ebezzam
#
#     Parameters
#     ----------
#     val : :py:class:`~numpy.ndarray`
#         Locations in meters.
#     cell_m : float
#         Dimension of cell in meters.
#     """
#     return np.array(val // cell_m, dtype=int)


def prepare_index_vals(key, pixel_pitch):
    """
    Convert indexing object in meters to indexing object in cell indices.

    author: Eric Bezzam,
    email: ebezzam@gmail.com,
    GitHub: https://github.com/ebezzam

    Parameters
    ----------
    key : int, float, slice, or list
        Indexing operation in meters.
    pixel_pitch : tuple(float)
        Pixel pitch (height, width) in meters.
    """

    if isinstance(key, (float, int)):
        idx = slice(None), _m_to_cell_idx(key, pixel_pitch[0])

    elif isinstance(key, slice):
        idx = slice(None), _cell_slice(key, pixel_pitch[0])

    elif len(key) == 2:
        idx = [slice(None)]
        for k, _slice in enumerate(key):

            if isinstance(_slice, slice):
                idx.append(_cell_slice(_slice, pixel_pitch[k]))

            elif isinstance(_slice, (float, int)):
                idx.append(_m_to_cell_idx(_slice, pixel_pitch[k]))

            else:
                raise ValueError("Invalid key.")
        idx = tuple(idx)

    elif len(key) == 3:
        raise NotImplementedError("Cannot index individual channels.")

    else:
        raise ValueError("Invalid key.")
    return idx


def rgb2gray(rgb, weights=None):
    """
    Convert RGB array to grayscale.

    author: Eric Bezzam,
    email: ebezzam@gmail.com,
    GitHub: https://github.com/ebezzam

    Parameters
    ----------
    rgb : :py:class:`~numpy.ndarray`
        (N_channel, N_height, N_width) image.
    weights : :py:class:`~numpy.ndarray`
        [Optional] (3,) weights to convert from RGB to grayscale.
    """
    if weights is None:
        weights = np.array([0.299, 0.587, 0.144])
    assert len(weights) == 3
    return np.tensordot(rgb, weights, axes=((0,), 0))


def extend_to_complex(angles):
    """
    Extend a tensor of angles into a complex tensor where the angles are used in
    the polar form for complex numbers and the respective magnitudes are set to
    1.

    Parameters
    ----------
    angles : torch.Tensor
        The tensor of angles to be used in the polar form

    Returns
    -------
    torch.Tensor
        The extended complex tensor
    """
    mags = torch.ones_like(angles)
    return torch.polar(mags, angles)


def load_holoeye_slm_pattern(
    path="images/holoeye_phase_map/holoeye_logo_slm_pattern.png",
):
    """
    Load a phase map generate with holoeye software and transform it into a
    compliant form.

    Parameters
    ----------
    path : str, optional
        The path to the phase map to load, by default
        "images/holoeye_phase_map/holoeye_logo_slm_pattern.png"

    Returns
    -------
    torch.Tensor
        The phase map transformed into a compliant form
    """
    im = Image.open(path)
    im = torch.from_numpy(np.array(im)).type(torch.FloatTensor)
    im = torch.mean(im, axis=2)

    max_val = torch.max(im)
    angles = (im / max_val) * (2 * np.pi) - np.pi

    holoeye_slm_field = extend_to_complex(angles)

    return holoeye_slm_field[None, None, :, :]


def show_plot(slm_field, propped_slm_field, title):
    """
    Plotting utility function.

    Parameters
    ----------
    slm_field : torch.Tensor
        The phase map before propagation
    propped_slm_field : torch.Tensor
        The amplitude after propagation
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
    ax1.imshow(slm_field.angle(), cmap="gray")
    ax2.imshow(slm_field.abs(), cmap="gray")
    ax3.imshow(propped_slm_field.angle(), cmap="gray")
    ax4.imshow(propped_slm_field.abs(), cmap="gray")
    plt.show()


def quantize_phase_pattern(phase_map):
    """
    Transform [-pi, pi] angles into the discrete interval 0-255.

    Parameters
    ----------
    phase_map : torch.Tensor or numpy.ndarray
        The angles to be quantized/discretized

    Returns
    -------
    numpy.ndarray
        The discretized map
    """
    if torch.is_tensor(phase_map):
        phase_map = phase_map.cpu().detach().numpy()

    if len(phase_map.shape):
        phase_map = phase_map[0, 0, :, :]

    phase_map += np.pi
    phase_map /= 2 * np.pi
    phase_map *= 255.0

    return np.rint(phase_map).astype("B")
