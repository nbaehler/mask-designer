import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from mask_designer.experimental_setup import amp_mask


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


# def si2cell(val: np.ndarray, cell_m): # TODO unused, remove
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


def load_image(path):
    """
    Load an image from a path.

    Parameters
    ----------
    path : String
        The path to the image

    Returns
    -------
    numpy.ndarray
        The image
    """
    img = Image.open(path)
    img = np.array(img)
    dtype = img.dtype

    if issubclass(dtype.type, np.floating):
        raise ValueError(
            "Problematic image type."
        )  # TODO check if this is correct, makes no sense when [0, 1]

    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]

        if img.shape[2] == 3:
            img = np.mean(img, axis=2)

    if issubclass(dtype.type, np.integer):
        img = img / np.iinfo(dtype).max

    # img = normalize_mask(img) # TODO needed?

    return round_phase_mask_to_uint8(img * 255)


def save_image(I, fname):
    """
    Save image to a file.

    Parameters
    ----------
    I : :py:class:`~numpy.ndarray`
        (N_channel, N_height, N_width) image.
    fname : str, path-like
        Valid image file (i.e. JPG, PNG, BMP, TIFF, etc.).
    """
    # I_max = I.max()   # TODO needed ?
    # I_max = 1 if np.isclose(I_max, 0) else I_max

    # I_f = I / I_max  # float64
    # I_u = np.uint8(255 * I_f)  # uint8

    # if I.ndim == 3:
    #     I = I.transpose(1, 2, 0)

    I_p = Image.fromarray(I)
    I_p.save(fname)


def load_field(path="images/phase_mask/holoeye_logo.png"):
    """
    Load a phase map, by default one generated with holoeye software, extends it
    to a field and transform it into a compliant form.

    Parameters
    ----------
    path : str, optional
        The path to the phase mask to load, by default
        "images/phase_mask/holoeye_logo.png"

    Returns
    -------
    torch.Tensor
        The field mask transformed into a compliant form
    """
    phase_mask = load_image(path)

    return extend_to_field(angularize_phase_mask(phase_mask))[None, None, :, :]


def extend_to_field(angles):
    """
    Extend angles into a field.
    """
    return torch.polar(amp_mask, angles)


def random_init_phase_mask(slm_shape, device, seed=1):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return (-0.5 + 1.0 * torch.rand(size=(1, 1, *slm_shape), generator=gen)).to(device)


def normalize_mask(mask):
    """
    Normalize the phase mask to be between 0 and 1.
    """
    if torch.is_tensor(mask):
        mask = mask.cpu().detach().numpy()

    if len(mask.shape) == 4:
        mask = mask[0, 0, :, :]

    minimum = np.min(mask)

    # maximum = np.max(mask)

    quantile = np.quantile(mask, 0.99)  # TODO do we need this?
    mask[mask > quantile] = quantile
    maximum = quantile

    return mask if minimum == maximum else (mask - minimum) / (maximum - minimum)


def angularize_phase_mask(phase_mask):
    if torch.is_tensor(phase_mask):
        phase_mask = phase_mask.cpu().detach().numpy()

    dtype = phase_mask.dtype

    if issubclass(dtype.type, np.floating):
        raise ValueError(
            "Problematic image type."
        )  # TODO check if this is correct, makes no sense when [0, 1]

    if len(phase_mask.shape) == 4:
        phase_mask = phase_mask[0, 0, :, :]

    max_value = np.iinfo(dtype).max

    # epsilon = 1e-6  # TODO how to handle the wrap around at -pi/pi when in [0,1
    # ]?
    # phase_mask = normalize_mask(phase_mask) # TODO needed?

    phase_mask = torch.from_numpy(phase_mask).type(torch.FloatTensor)
    return (phase_mask / max_value) * (2 * np.pi) - np.pi


def quantize_phase_mask(phase_mask):
    """
    Transform [-pi, pi] angles into the discrete interval 0-255.

    Parameters
    ----------
    phase_mask : torch.Tensor or numpy.ndarray
        The angles to be quantized/discretized

    Returns
    -------
    numpy.ndarray
        The discretized map
    """
    if torch.is_tensor(phase_mask):
        phase_mask = phase_mask.cpu().detach().numpy()

    if len(phase_mask.shape) == 4:
        phase_mask = phase_mask[0, 0, :, :]

    new_phase_mask = phase_mask + np.pi
    new_phase_mask /= 2 * np.pi

    # new_phase_mask = normalize_mask(new_phase_mask) # TODOneeded?

    new_phase_mask *= 255.0

    return round_phase_mask_to_uint8(new_phase_mask)


def round_phase_mask_to_uint8(phase_mask):
    """
    Round the phase_mask to the nearest integer and then convert to uint8.
    """
    return np.round(phase_mask).astype(np.uint8)


def scale_image_to_shape(image, shape, pad=False):
    dtype = image.dtype

    # Height / Width
    aspect_ratio_orig = image.shape[0] / image.shape[1]
    aspect_ratio_target = shape[0] / shape[1]

    if aspect_ratio_orig != aspect_ratio_target:
        if (
            aspect_ratio_orig < aspect_ratio_target
            and pad
            or aspect_ratio_orig > aspect_ratio_target
            and not pad
        ):
            target_shape = (round(image.shape[1] * aspect_ratio_target), image.shape[1])
        elif aspect_ratio_orig < aspect_ratio_target or aspect_ratio_orig > aspect_ratio_target:
            target_shape = (image.shape[0], round(image.shape[0] / aspect_ratio_target))
        if pad:
            image = pad_image_to_shape(image, target_shape)
        else:
            image = crop_image_to_shape(image, target_shape)

    im = Image.fromarray(image)
    im = im.resize((shape[1], shape[0]), Image.BICUBIC)  # Pillow uses width, height
    return np.array(im).astype(dtype)


def crop_image_to_shape(image, shape):
    top = (image.shape[0] - shape[0]) // 2
    bottom = image.shape[0] - shape[0] - top
    left = (image.shape[1] - shape[1]) // 2
    right = image.shape[1] - shape[1] - left

    return image[
        top : image.shape[0] - bottom, left : image.shape[1] - right,
    ]


def pad_image_to_shape(image, shape):
    top = (shape[0] - image.shape[0]) // 2
    bottom = shape[0] - image.shape[0] - top
    left = (shape[1] - image.shape[1]) // 2
    right = shape[1] - image.shape[1] - left

    pad_shape = ((top, bottom), (left, right))

    return np.pad(image, pad_shape)


def pad_tensor_to_shape(phase_map, shape):
    # that you remove and add dims back again where needed for this function
    top = (shape[0] - phase_map.shape[2]) // 2
    bottom = shape[0] - phase_map.shape[2] - top
    left = (shape[1] - phase_map.shape[3]) // 2
    right = shape[1] - phase_map.shape[3] - left

    pad_shape = (left, right, top, bottom)

    return F.pad(phase_map, pad_shape, "constant", 0)
