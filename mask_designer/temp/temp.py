import numpy as np
from PIL import Image

from mask_designer.experimental_setup import slm_device

from slm_controller.hardware import (
    SLMParam,
    slm_devices,
)

from mask_designer.utils import resize_image_to_shape


def main():
    slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]

    image = Image.open("images/target_amplitude/holoeye_logo.png")
    image = np.array(image)

    resized_image = resize_image_to_shape(image, slm_shape, pad=True)
    resized_image = Image.fromarray(resized_image)

    resized_image.save("images/test/holoeye_logo_big.png")
