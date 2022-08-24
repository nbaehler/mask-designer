from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)


import numpy as np
import torch
from PIL import Image

from mask_designer.utils import (
    build_field,
    quantize_phase_mask,
    angularize_phase_mask,
)


def main():
    im = Image.open("images/holoeye_phase_mask/holoeye_logo.png")
    phase_mask = torch.from_numpy(np.array(im)).type(torch.FloatTensor)

    if len(phase_mask.shape) == 3:
        if phase_mask.shape[2] == 4:
            phase_mask = phase_mask[:, :, :3]

        if phase_mask.shape[2] == 3:
            phase_mask = torch.mean(phase_mask, axis=2)

    v1 = build_field(angularize_phase_mask(phase_mask))
    v2 = v1.angle()
    mags = torch.ones_like(v2)
    v3 = torch.polar(mags, v2)
    v4 = build_field(v2)
    v5 = quantize_phase_mask(v2)

    print(torch.min(phase_mask), torch.max(phase_mask))
    print(torch.min(v2), torch.max(v2))
    print(np.min(v5), np.max(v5))

    print(torch.eq(mags, v3.abs()).all())
    print(torch.eq(v2, v3.angle()).all())
    print(torch.eq(v1.angle(), v2).all())

    print(torch.eq(v3, v4).all())
    print(torch.eq(v4, v1).all())

    # Max value of input is 254, not 255, so this might be false
    print(torch.eq(phase_mask, torch.from_numpy(v5)).all())

    # for i in range(phase_mask.shape[0]):
    #     for j in range(phase_mask.shape[1]):
    #         if phase_mask[i, j] != v3[i, j]:
    #             print(i, j, phase_mask[i, j], v3[i, j])


if __name__ == "__main__":
    main()
