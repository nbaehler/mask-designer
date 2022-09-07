import abc

import numpy as np
from ids_peak import ids_peak as peak
from ids_peak_ipl import ids_peak_ipl as peak_ipl
from slm_controller.hardware import SLMParam, slm_devices

from mask_designer.experimental_setup import slm_device
from mask_designer.hardware import CamDevices, CamParam, cam_devices
from mask_designer.utils import load_image, scale_image_to_shape


class Camera:
    def __init__(self):
        """
        Abstract class capturing the functionalities of the cameras used in this project.
        """
        self._width = -1
        self._height = -1
        self._frame_count = 0
        self._exposure_time = -1
        self._correction = None

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def shape(self):
        return (self._height, self._width)

    @property
    def frame(self):
        return self._frame_count

    def acquire_multiple_images_and_resize_to_slm_shape(self, number=2):
        """
        Triggers the acquisition of multiple images and then resizes them to the size
        of the SLM.

        Parameters
        ----------
        number : int, optional
            The number of images taken, by default 2

        Returns
        -------
        list
            The list of the acquired images that are resized to match the slm shape
        """
        images = self.acquire_multiple_images(number)

        return [
            scale_image_to_shape(image, slm_devices[slm_device][SLMParam.SLM_SHAPE])
            for image in images
        ]

    def acquire_single_image_and_resize_to_slm_shape(self):
        """
        Triggers the acquisition of a single images and then resizes it to the size
        of the SLM.

        Returns
        -------
        ndarray
            The acquired image that is resized to match the slm shape
        """
        image = self.acquire_single_image()

        return scale_image_to_shape(image, slm_devices[slm_device][SLMParam.SLM_SHAPE])

    @abc.abstractmethod
    def acquire_single_image(self):
        """
        Triggers the acquisition of a single image.
        """
        pass

    @abc.abstractmethod
    def acquire_multiple_images(self, number=2):
        """
        Triggers the acquisition of multiple images.

        Parameters
        ----------
        number : int, optional
            The number of images taken, by default 2
        """
        pass

    @abc.abstractmethod
    def set_exposure_time(self, time):
        """
        Set the exposure time of the camera to a specific value.

        Parameters
        ----------
        time : int
            New exposure time
        """
        pass

    def set_correction(
        self,
        correction=np.zeros(cam_devices[CamDevices.DUMMY.value][CamParam.SHAPE], dtype=np.uint8),
    ):
        """
        Set the correction of the camera to a specific value.

        Parameters
        ----------
        correction : ndarray
            New correction
        """

        if correction.shape != cam_devices[CamDevices.DUMMY.value][CamParam.SHAPE]:
            raise ValueError("The correction must have the same shape as the camera image")

        self._correction = correction


class DummyCamera(Camera):
    def __init__(self):
        """
        Initializes the dummy camera returning only black images. # TODO update comments/documentation
        """
        super().__init__()

        # Set height and width
        self._height, self._width = cam_devices[CamDevices.DUMMY.value][CamParam.SHAPE]

        # Set frame count and exposure time
        self.set_exposure_time()

        self.set_correction()

        # Default image is just a white image
        self._image = (
            np.ones((self._height, self._width), dtype=np.uint8) * 255
        )  # TODO change comments/documentation to white image (all 255)

    def set_exposure_time(self, time=np.pi):
        """
        Set the exposure time of the camera to a specific value.

        Parameters
        ----------
        time : int, optional
            New exposure time, by default pi
        """
        self._exposure_time = time

    def use_image(self, path="citl/calibration/cali.png"):  # TODO documentation
        image = load_image(path) - self._correction
        self._image = scale_image_to_shape(image, (self._height, self._width))

    def acquire_single_image(self):
        """
        Acquire a single dummy image.

        Returns
        -------
        ndarray
            The acquired dummy image
        """

        self._frame_count += 1
        return self._image.copy()

    def acquire_multiple_images(self, number=2):
        """
        Acquire multiple dummy images.

        Parameters
        ----------
        number : int, optional
            The number of images to be taken, by default 2

        Returns
        -------
        list
            The list of acquired dummy images
        """
        self._frame_count += number
        return [self._image.copy() for _ in range(number)]


class IDSCamera(Camera):
    # \file    mainwindow.py # TODO inspired by this, license??
    # \author  IDS Imaging Development Systems GmbH
    # \date    2021-01-15
    # \since   1.2.0
    #
    # \version 1.1.1
    #
    # Copyright (C) 2021, IDS Imaging Development Systems GmbH.
    #
    # The information in this document is subject to change without notice
    # and should not be construed as a commitment by IDS Imaging Development Systems GmbH.
    # IDS Imaging Development Systems GmbH does not assume any responsibility for any errors
    # that may appear in this document.
    #
    # This document, or source code, is provided solely as an example of how to utilize
    # IDS Imaging Development Systems GmbH software libraries in a sample application.
    # IDS Imaging Development Systems GmbH does not assume any responsibility
    # for the use or reliability of any portion of this document.
    #
    # General permission to copy or modify is hereby granted.

    def __init__(self):
        """
        Initializes the camera and sets all the parameters such that acquisition
        in out setting can be started right away.

        Raises
        ------
        IOError
            If no compatible devices cameras are detected by the system
        """
        super().__init__()

        # Set height and width
        self._height, self._width = cam_devices[CamDevices.IDS.value][CamParam.SHAPE]

        # Initialize library
        peak.Library.Initialize()

        # Create a device manager object
        device_manager = peak.DeviceManager.Instance()

        self.__data_stream = None

        # Update device manager
        try:
            device_manager.Update()
        except:
            raise IOError(
                "Failed to update the IDS device manager. Check the cameras connection and setup."
            )

        # Raise exception if no device was found
        if device_manager.Devices().empty():
            raise IOError(
                "Failed to load IDS camera in the device manager. Check its connection and setup."
            )

        self.__device = None

        # Open the first openable device in the managers device list
        for device in device_manager.Devices():
            if device.IsOpenable():
                self.__device = device.OpenDevice(peak.DeviceAccessType_Control)
                break

        if self.__device is None:
            raise IOError("Failed to open IDS camera. Check its connection and setup.")

        # Get nodemap of the remote device for all accesses to the genicam nodemap tree
        self.__node_map = self.__device.RemoteDevice().NodeMaps()[0]

        # Load default settings
        self.__node_map.FindNode("UserSetSelector").SetCurrentEntry("Default")
        self.__node_map.FindNode("UserSetLoad").Execute()
        self.__node_map.FindNode("UserSetLoad").WaitUntilDone()

        # Use single frame mode
        self.__node_map.FindNode("AcquisitionMode").SetCurrentEntry("SingleFrame")

        # https://de.ids-imaging.com/manuals/ids-peak/ids-peak-user-manual/1.3.0/en/operate-single-frame-acquisition.html?q=SingleFrame
        self.__node_map.FindNode("TriggerSelector").SetCurrentEntry("ExposureStart")
        self.__node_map.FindNode("TriggerMode").SetCurrentEntry("Off")

        # Lock critical features to prevent them from changing
        self.__node_map.FindNode("TLParamsLocked").SetValue(1)

        data_streams = self.__device.DataStreams()
        if data_streams.empty():
            # no data streams available
            raise IOError(
                "Failed to access the IDS camera data streams. Check its connection and setup."
            )

        # Open standard data stream
        self.__data_stream = data_streams[0].OpenDataStream()

        self.__flush_and_revoke_buffers()
        self.__allocate_buffers()

        self.set_correction()

        # Set the exposure time to default value
        self.set_exposure_time()

    def __allocate_buffers(self):
        if self.__data_stream:
            payload_size = self.__node_map.FindNode("PayloadSize").Value()

            # Get number of minimum required buffers
            num_buffers_min_required = self.__data_stream.NumBuffersAnnouncedMinRequired()

            # Allocate buffers
            for _ in range(num_buffers_min_required):
                buffer = self.__data_stream.AllocAndAnnounceBuffer(payload_size)
                self.__data_stream.QueueBuffer(buffer)

    def __flush_and_revoke_buffers(self):
        if self.__data_stream:
            # Flush queue and prepare all buffers for revoking
            self.__data_stream.Flush(peak.DataStreamFlushMode_DiscardAll)

            # Clear all old buffers
            for buffer in self.__data_stream.AnnouncedBuffers():
                self.__data_stream.RevokeBuffer(buffer)

    def __single_acquisition(self):
        # Start acquisition on camera
        acquisition_start = self.__node_map.FindNode("AcquisitionStart")
        self.__data_stream.StartAcquisition()
        acquisition_start.Execute()
        acquisition_start.WaitUntilDone()

        # Get data from device's data stream, in milliseconds
        buffer = self.__data_stream.WaitForFinishedBuffer(5000)

        # Stop acquisition on camera
        self.__data_stream.StopAcquisition()

        # Flush data stream
        self.__data_stream.Flush(peak.DataStreamFlushMode_DiscardAll)

        # Queue buffer so that it can be used again
        self.__data_stream.QueueBuffer(buffer)

        return buffer

    def __print_supported_nodes(self):  # for development
        all_nodes = self.__node_map.Nodes()

        available_nodes = [
            node.DisplayName()
            for node in all_nodes
            if node.AccessStatus()
            not in [peak.NodeAccessStatus_NotAvailable, peak.NodeAccessStatus_NotImplemented,]
        ]

        available_nodes.sort()

        for node in available_nodes:
            print(node)

    def __print_supported_entries(self, node):  # for development
        all_entries = self.__node_map.FindNode(node).Entries()
        available_entries = [
            entry.SymbolicValue()
            for entry in all_entries
            if entry.AccessStatus()
            not in [peak.NodeAccessStatus_NotAvailable, peak.NodeAccessStatus_NotImplemented,]
        ]

        available_entries.sort()

        for entry in available_entries:
            print(entry)

    def set_exposure_time(self, time=33.189):
        """
        Set the exposure time of the camera to a specific value.

        Parameters
        ----------
        time : int, optional
            New exposure time in microseconds, by default 33.189 (which is the
            minimal value supported by the camera)
        """
        # Store the new exposure time
        self._exposure_time = time

        # Unlock the parameters lock
        self.__node_map.FindNode("TLParamsLocked").SetValue(0)

        # Change exposure time (in microseconds)
        self.__node_map.FindNode("ExposureTime").SetValue(float(time))

        # Lock critical features to prevent them from changing
        self.__node_map.FindNode("TLParamsLocked").SetValue(1)

        # Hacky fix for a bug with exposure time
        self.__single_acquisition()  # Bug fix, parameters are only committed after a capture

    def acquire_single_image(self):
        """
        Acquire a single image.

        Returns
        -------
        ndarray
            Acquired image
        """
        self._frame_count += 1

        # Perform a single image acquisition
        buffer = self.__single_acquisition()

        # Create IDS peak IPL image
        ipl_image = peak_ipl.Image_CreateFromSizeAndBuffer(
            buffer.PixelFormat(), buffer.BasePtr(), buffer.Size(), self._width, self._height,
        )

        # Return image as numpy array
        return np.flipud(np.fliplr(ipl_image.get_numpy_2D().copy())) - self._correction

    def acquire_multiple_images(self, number=2):
        """
        Acquire multiple images.

        Parameters
        ----------
        number : int, optional
            The number of images to be taken, by default 2

        Returns
        -------
        list
            The list of acquired images
        """
        images = []
        self._frame_count += number

        for _ in range(number):
            buffer = self.__single_acquisition()

            # Create IDS peak IPL image
            ipl_image = peak_ipl.Image_CreateFromSizeAndBuffer(
                buffer.PixelFormat(), buffer.BasePtr(), buffer.Size(), self._width, self._height,
            )

            # Append images as numpy array to list of acquired images
            images.append(
                np.flipud(np.fliplr(ipl_image.get_numpy_2D().copy())) - self._correction
            )  # TODO really need to flip?

        return images

    def __del__(self):
        """
        Clean up before exit.
        """

        self.__flush_and_revoke_buffers()

        peak.Library.Close()


def create(device_key):
    """
    Factory method to create `Camera` object.

    Parameters
    ----------
    device_key : str
        Option from `CamDevices`.
    """
    assert device_key in CamDevices.values()

    cam_device = None
    if device_key == CamDevices.DUMMY.value:
        cam_device = DummyCamera()
    elif device_key == CamDevices.IDS.value:
        cam_device = IDSCamera()

    assert cam_device is not None
    return cam_device
