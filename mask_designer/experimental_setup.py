from enum import Enum

import torch
from skimage.draw import disk
from slm_controller.hardware import SLMDevices, SLMParam, slm_devices

from mask_designer.hardware import CamDevices


# Parameters relevant for the experiments
class Params(Enum):
    WAVELENGTH = "wavelength"
    PROPAGATION_DISTANCE = "prop_distance"
    ROI = "roi"

    SLM_SETTLE_TIME = "slm_settle_time"
    SLM_SHOW_TIME = "slm_show_time"

    ITERATIONS = "iterations"
    WARM_START_ITERATIONS = "warm_start_iterations"
    CITL_ITERATIONS = "citl_iterations"

    @staticmethod
    def values():
        return [param.value for param in Params]


# Default values of those parameters
default_params = {
    Params.WAVELENGTH: 532e-9,
    Params.PROPAGATION_DISTANCE: 0.275,
    Params.ROI: (640, 880),
    Params.SLM_SETTLE_TIME: 0.25,
    Params.SLM_SHOW_TIME: 10,
    Params.ITERATIONS: 500,
    Params.WARM_START_ITERATIONS: 100,
    Params.CITL_ITERATIONS: 10,
}

# Choose a slm device
slm_device = SLMDevices.HOLOEYE_LC_2012.value

# and a camera device that you want to use
cam_device = CamDevices.IDS.value


def circular_amp():  # From here https://stackoverflow.com/a/70283438
    # Laser radius = 1cm := r
    # pixel pitch = 0.36e-4 m = 0.0036 cm,
    # ==> r = 278 px

    shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]

    amp = torch.zeros(shape)

    center = (shape[0] / 2, shape[1] / 2)
    radius = 278
    rr, cc = disk(center, radius)
    amp[rr, cc] = 1

    return amp


def rectangular_amp():
    return torch.ones(slm_devices[slm_device][SLMParam.SLM_SHAPE])


# Chose the shape of the laser beam hitting the SLM
amp_mask = circular_amp()
