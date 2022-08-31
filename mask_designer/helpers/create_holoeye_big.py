from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)


import numpy as np
from PIL import Image

from slm_controller.hardware import (
    SLMParam,
    slm_devices,
)

from mask_designer.experimental_setup import slm_device
from mask_designer.utils import scale_image_to_shape


def main():
    slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]

    image = Image.open("images/target_amplitude/holoeye_logo.png")
    image = np.array(image)

    resized_image = scale_image_to_shape(image, slm_shape, pad=True)
    resized_image = Image.fromarray(resized_image)

    resized_image.save("images/test/holoeye_logo_big.png")


if __name__ == "__main__":
    main()
