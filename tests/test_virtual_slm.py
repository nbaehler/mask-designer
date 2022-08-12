"""
author: Eric Bezzam,
email: ebezzam@gmail.com,
GitHub: https://github.com/ebezzam
"""

from mask_designer.virtual_slm import VirtualSLM
import numpy as np

dim = 10
s = VirtualSLM(shape=(dim, dim), pixel_pitch=(1, 1))


class TestVirtualSLM:
    """
    Test :py:module:`~mask_designer.virtual_slm.VirtualSLM`.
    """

    def test_row_indexing(self):
        # single row
        val = s.at(3.5)
        assert val.shape == (3, dim)
        val = s.at(np.s_[3.5, :])

        assert val.shape == (3, dim)

        # get a few rows
        val = s.at(np.s_[1.5:4])
        assert val.shape == (3, 3, dim)
        val = s.at(np.s_[:4.5])
        assert val.shape == (3, 4, dim)
        val = s.at(np.s_[4.5:])
        assert val.shape == (3, 6, dim)

    def test_col_indexing(self):
        # single column
        val = s.at(np.s_[:, 3.5])
        assert val.shape == (3, dim)

        # get a few columns
        val = s.at(np.s_[:, 1.5:4])
        assert val.shape == (3, dim, 3)
        val = s.at(np.s_[:, :4.5])
        assert val.shape == (3, dim, 4)
        val = s.at(np.s_[:, 4.5:])
        assert val.shape == (3, dim, 6)

    def test_grayscale_values(self):
        vals = s.grayscale_values
        assert vals.shape == (10, 10)
