from enum import Enum

# Physical parameters relevant for the propagation
class PhysicalParams(Enum):  # TODO better place?
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
