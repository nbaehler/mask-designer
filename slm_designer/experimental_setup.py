from slm_controller.hardware import SLMDevices
from slm_designer.hardware import CamDevices

from enum import Enum

# Physical parameters relevant for the propagation
class PhysicalParams(Enum):
    WAVELENGTH = "wavelength"
    PROPAGATION_DISTANCE = "prop_distance"

    @staticmethod
    def values():
        return [param.value for param in PhysicalParams]


# Actual values of those physical parameters
physical_params = {
    PhysicalParams.WAVELENGTH: 532e-9,
    PhysicalParams.PROPAGATION_DISTANCE: 0.34,
}

# Choose slm and camera that you want to use
slm_device = SLMDevices.HOLOEYE_LC_2012.value
cam_device = CamDevices.IDS.value
