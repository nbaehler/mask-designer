from slm_controller.hardware import SLMDevices
from slm_designer.hardware import CamDevices

from enum import Enum

# Physical parameters relevant for the propagation
class PhysicalParams(Enum):
    WAVELENGTH = "wavelength"
    PROPAGATION_DISTANCE = "prop_distance"
    SLM_SETTLE_TIME = "slm_settle_time"

    @staticmethod
    def values():
        return [param.value for param in PhysicalParams]


# Actual values of those physical parameters
physical_params = {
    PhysicalParams.WAVELENGTH: 532e-9,
    PhysicalParams.PROPAGATION_DISTANCE: 0.34,
    PhysicalParams.SLM_SETTLE_TIME: 0.25,
}

# Choose a slm device
slm_device = SLMDevices.HOLOEYE_LC_2012.value

# and a camera device that you want to use
# cam_device = CamDevices.IDS.value
cam_device = CamDevices.DUMMY.value
