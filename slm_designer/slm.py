"""
author: Eric Bezzam,
email: ebezzam@gmail.com,
GitHub: https://github.com/ebezzam
"""


import numpy as np
from slm_designer.utils import prepare_index_vals, rgb2gray
import matplotlib.pyplot as plt


class SLM:
    def __init__(self, shape, pixel_pitch):
        """
        Class for defining SLM.

        Parameters
        ----------
        shape : tuple(int)
            (height, width) in number of cell.
        pixel_pitch : tuple(float)
            Pixel pitch (height, width) in meters.
        """
        assert np.all(shape) > 0
        assert np.all(pixel_pitch) > 0
        self._shape = shape
        self._pixel_pitch = pixel_pitch
        self._values = np.zeros((3,) + shape)

    @property
    def size(self):
        return np.prod(self._shape)

    @property
    def shape(self):
        return self._shape

    @property
    def pixel_pitch(self):
        return self._pixel_pitch

    @property
    def center(self):
        return np.array([self.height / 2, self.width / 2])

    @property
    def dim(self):
        return np.array(self._shape) * np.array(self._pixel_pitch)

    @property
    def height(self):
        return self.dim[0]

    @property
    def width(self):
        return self.dim[1]

    @property
    def values(self):
        return self._values

    @property
    def grayscale_values(self):
        return rgb2gray(self._values)

    def at(self, physical_coord, value=None):
        """
        Get/set values of SLM at physical coordinate in meters.

        Parameters
        ----------
        physical_coord : int, float, slice tuples
            Physical coordinates to get/set SLM values.
        value : int, float, :py:class:`~numpy.ndarray`
            [Optional] values to set, otherwise return values at specified coordinates.
        """
        idx = prepare_index_vals(physical_coord, self._pixel_pitch)
        if value is None:
            # getter
            return self._values[idx]
        else:
            # setter
            self._values[idx] = value

    def __getitem__(self, key):
        return self._values[key]

    def __setitem__(self, key, value):
        self._values[key] = value

    def plot(self, show_tick_labels=False):
        """
        Plot SLM pattern.

        Parameters
        ----------
        show_tick_labels : bool
            Whether to show cell number along x- and y-axis.
        """

        # prepare mask data for `imshow`, expects the input data array size to be (width, height, 3)
        Z = self.values.transpose(1, 2, 0)

        # plot
        _, ax = plt.subplots()
        extent = [
            -0.5 * self._pixel_pitch[1],
            (self._shape[1] - 0.5) * self._pixel_pitch[1],
            (self._shape[0] - 0.5) * self._pixel_pitch[0],
            -0.5 * self._pixel_pitch[0],
        ]
        ax.imshow(Z, extent=extent)
        ax.grid(which="major", axis="both", linestyle="-", color="0.5", linewidth=0.25)

        x_ticks = np.arange(-0.5, self._shape[1], 1) * self._pixel_pitch[1]
        ax.set_xticks(x_ticks)
        if show_tick_labels:
            x_tick_labels = (np.arange(-0.5, self._shape[1], 1) + 0.5).astype(int)
        else:
            x_tick_labels = [None] * len(x_ticks)
        ax.set_xticklabels(x_tick_labels)

        y_ticks = np.arange(-0.5, self._shape[0], 1) * self._pixel_pitch[0]
        ax.set_yticks(y_ticks)
        if show_tick_labels:
            y_tick_labels = (np.arange(-0.5, self._shape[0], 1) + 0.5).astype(int)
        else:
            y_tick_labels = [None] * len(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        return ax
