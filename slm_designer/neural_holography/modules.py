"""
https://github.com/computational-imaging/neural-holography/blob/d2e399014aa80844edffd98bca34d2df80a69c84/utils/modules.py

Some modules for easy use. (No need to calculate kernels explicitly)

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.

Copyright (c) 2020, Stanford University

All rights reserved.

Refer to the LICENSE file for more information.
"""

import numpy as np
import torch
import torch.nn as nn
from slm_controller import slm
from slm_designer import camera

from slm_designer.neural_holography.algorithms import (
    gerchberg_saxton,
    stochastic_gradient_descent,
    double_phase_amplitude_coding,
)

import os
import time
import skimage.io
import slm_designer.neural_holography.utils as utils
from slm_designer.neural_holography.propagation_ASM import propagation_ASM
from slm_designer.neural_holography.calibration_module import Calibration

import platform

# my_os = platform.system()
# if my_os == "Windows":
#     from slm_designer.neural_holography.arduino_laser_control_module import (
#         ArduinoLaserControl,
#     )
#     from slm_designer.neural_holography.camera_capture_module import CameraCapture
#     from slm_designer.neural_holography.calibration_module import Calibration
#     from slm_designer.neural_holography.slm_display_module import SLMDisplay


class GS(nn.Module):
    """Classical Gerchberg-Saxton algorithm

    Class initialization parameters
    -------------------------------
    :param prop_dist: propagation dist between SLM and target, in meters
    :param wavelength: the wavelength of interest, in meters
    :param feature_size: the SLM pixel pitch, in meters
    :param num_iters: the number of iteration, default 500
    :param phase_path: path to write intermediate results
    :param loss: loss function, default L2
    :param prop_model: chooses the propagation operator ('ASM': propagation_ASM,
        'model': calibrated model). Default 'ASM'.
    :param propagator: propagator instance (function / pytorch module)
    :param device: torch.device

    Usage
    -----
    Functions as a pytorch module:

    >>> gs = GS(...)
    >>> final_phase = gs(target_amp, init_phase)

    target_amp: amplitude at the target plane, with dimensions [batch, 1, height, width]
    init_phase: initial guess of phase of phase-only slm
    final_phase: optimized phase-only representation at SLM plane, same dimensions
    """

    def __init__(
        self,
        prop_dist,
        wavelength,
        feature_size,
        num_iters,
        phase_path=None,
        prop_model="ASM",
        propagator=propagation_ASM,
        writer=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(GS, self).__init__()

        # Setting parameters
        self.prop_dist = prop_dist
        self.wavelength = wavelength
        self.feature_size = feature_size
        self.phase_path = phase_path
        self.precomputed_H_f = None
        self.precomputed_H_b = None
        self.prop_model = prop_model
        self.prop = propagator
        self.num_iters = num_iters
        self.writer = writer
        self.dev = device

    def forward(self, target_amp, init_phase=None):
        # Pre-compute propagation kernel only once
        if self.precomputed_H_f is None and self.prop_model == "ASM":
            self.precomputed_H_f = self.prop(
                torch.empty(*init_phase.shape, dtype=torch.complex64),
                self.feature_size,
                self.wavelength,
                self.prop_dist,
                return_H=True,
            )
            self.precomputed_H_f = self.precomputed_H_f.to(self.dev).detach()
            self.precomputed_H_f.requires_grad = False

        if self.precomputed_H_b is None and self.prop_model == "ASM":
            self.precomputed_H_b = self.prop(
                torch.empty(*init_phase.shape, dtype=torch.complex64),
                self.feature_size,
                self.wavelength,
                -self.prop_dist,
                return_H=True,
            )
            self.precomputed_H_b = self.precomputed_H_b.to(self.dev).detach()
            self.precomputed_H_b.requires_grad = False

        # Run algorithm
        return gerchberg_saxton(
            init_phase,
            target_amp,
            self.num_iters,
            self.prop_dist,
            self.wavelength,
            self.feature_size,
            # phase_path=self.phase_path,
            prop_model=self.prop_model,
            propagator=self.prop,
            precomputed_H_f=self.precomputed_H_f,
            precomputed_H_b=self.precomputed_H_b,
            # writer=self.writer,
        )

    @property
    def phase_path(self):
        return self._phase_path

    @phase_path.setter
    def phase_path(self, phase_path):
        self._phase_path = phase_path


class SGD(nn.Module):
    """Proposed Stochastic Gradient Descent Algorithm using Auto-diff Function of PyTorch

    Class initialization parameters
    -------------------------------
    :param prop_dist: propagation dist between SLM and target, in meters
    :param wavelength: the wavelength of interest, in meters
    :param feature_size: the SLM pixel pitch, in meters
    :param num_iters: the number of iteration, default 500
    :param roi_res: region of interest to penalize the loss
    :param phase_path: path to write intermediate results
    :param loss: loss function, default L2
    :param prop_model: chooses the propagation operator ('ASM': propagation_ASM,
        'model': calibrated model). Default 'ASM'.
    :param propagator: propagator instance (function / pytorch module)
    :param lr: learning rate for phase variables
    :param lr_s: learning rate for the learnable scale
    :param s0: initial scale
    :param writer: SummaryWrite instance for tensorboard
    :param device: torch.device

    Usage
    -----
    Functions as a pytorch module:

    >>> sgd = SGD(...)
    >>> final_phase = sgd(target_amp, init_phase)

    target_amp: amplitude at the target plane, with dimensions [batch, 1, height, width]
    init_phase: initial guess of phase of phase-only slm
    final_phase: optimized phase-only representation at SLM plane, same dimensions
    """

    def __init__(
        self,
        prop_dist,
        wavelength,
        feature_size,
        num_iters,
        roi_res,
        phase_path=None,
        prop_model="ASM",
        propagator=propagation_ASM,
        loss=nn.MSELoss(),
        lr=0.01,
        lr_s=0.003,
        s0=1.0,
        citl=False,
        camera_prop=None,
        writer=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(SGD, self).__init__()

        # Setting parameters
        self.prop_dist = prop_dist
        self.wavelength = wavelength
        self.feature_size = feature_size
        self.roi_res = roi_res
        self.phase_path = phase_path
        self.precomputed_H = None
        self.prop_model = prop_model
        self.prop = propagator

        self.num_iters = num_iters
        self.lr = lr
        self.lr_s = lr_s
        self.init_scale = s0

        self.citl = citl
        self.camera_prop = camera_prop

        self.writer = writer
        self.dev = device
        self.loss = loss.to(device)

    def forward(self, target_amp, init_phase=None):
        # Pre-compute propagation kernel only once
        if self.precomputed_H is None and self.prop_model == "ASM":
            self.precomputed_H = self.prop(
                torch.empty(*init_phase.shape, dtype=torch.complex64),
                self.feature_size,
                self.wavelength,
                self.prop_dist,
                return_H=True,
            )
            self.precomputed_H = self.precomputed_H.to(self.dev).detach()
            self.precomputed_H.requires_grad = False

        # Run algorithm
        return stochastic_gradient_descent(
            init_phase,
            target_amp,
            self.num_iters,
            self.prop_dist,
            self.wavelength,
            self.feature_size,
            roi_res=self.roi_res,
            phase_path=self.phase_path,
            prop_model=self.prop_model,
            propagator=self.prop,
            loss=self.loss,
            lr=self.lr,
            lr_s=self.lr_s,
            s0=self.init_scale,
            citl=self.citl,
            camera_prop=self.camera_prop,
            writer=self.writer,
            precomputed_H=self.precomputed_H,
        )

    @property
    def init_scale(self):
        return self._init_scale

    @init_scale.setter
    def init_scale(self, s):
        self._init_scale = s

    @property
    def citl_hardwares(self):
        return self._citl_hardwares

    @citl_hardwares.setter
    def citl_hardwares(self, citl_hardwares):
        self._citl_hardwares = citl_hardwares

    @property
    def phase_path(self):
        return self._phase_path

    @phase_path.setter
    def phase_path(self, phase_path):
        self._phase_path = phase_path

    @property
    def prop(self):
        return self._prop

    @prop.setter
    def prop(self, prop):
        self._prop = prop


class DPAC(nn.Module):
    """Double-phase Amplitude Coding

    Class initialization parameters
    -------------------------------
    :param prop_dist: propagation dist between SLM and target, in meters
    :param wavelength: the wavelength of interest, in meters
    :param feature_size: the SLM pixel pitch, in meters
    :param prop_model: chooses the propagation operator ('ASM': propagation_ASM,
        'model': calibrated model). Default 'ASM'.
    :param propagator: propagator instance (function / pytorch module)
    :param device: torch.device

    Usage
    -----
    Functions as a pytorch module:

    >>> dpac = DPAC(...)
    >>> final_phase = dpac(target_amp, target_phase)

    target_amp: amplitude at the target plane, with dimensions [batch, 1, height, width]
    target_amp (optional): phase at the target plane, with dimensions [batch, 1, height, width]
    final_phase: optimized phase-only representation at SLM plane, same dimensions

    """

    def __init__(
        self,
        prop_dist,
        wavelength,
        feature_size,
        prop_model="ASM",
        propagator=propagation_ASM,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(DPAC, self).__init__()

        # propagation is from target to SLM plane (one step)
        self.prop_dist = -prop_dist
        self.wavelength = wavelength
        self.feature_size = feature_size
        self.precomputed_H = None
        self.prop_model = prop_model
        self.prop = propagator
        self.dev = device

    def forward(self, target_amp, target_phase=None):
        if target_phase is None:
            target_phase = torch.zeros_like(target_amp)

        if self.precomputed_H is None and self.prop_model == "ASM":
            self.precomputed_H = self.prop(
                torch.empty(*target_amp.shape, dtype=torch.complex64),
                self.feature_size,
                self.wavelength,
                self.prop_dist,
                return_H=True,
            )
            self.precomputed_H = self.precomputed_H.to(self.dev).detach()
            self.precomputed_H.requires_grad = False

        return double_phase_amplitude_coding(
            target_phase,
            target_amp,
            self.prop_dist,
            self.wavelength,
            self.feature_size,
            prop_model=self.prop_model,
            propagator=self.prop,
            precomputed_H=self.precomputed_H,
        )

    @property
    def phase_path(self):
        return self._phase_path

    @phase_path.setter
    def phase_path(self, phase_path):
        self._phase_path = phase_path


class PhysicalProp(nn.Module):
    """A module for physical propagation,
    forward pass displays gets SLM pattern as an input and display the pattern on the physical setup,
    and capture the diffraction image at the target plane,
    and then return warped image using pre-calibrated homography from instantiation.

    Class initialization parameters
    -------------------------------
    :param channel:
    :param slm_settle_time:
    :param roi_res:
    :param num_circles:
    :param laser_arduino:
    :param com_port:
    :param arduino_port_num:
    :param range_row:
    :param range_col:
    :param patterns_path:
    :param calibration_preview:

    Usage
    -----
    Functions as a pytorch module:

    >>> camera_prop = PhysicalProp(...)
    >>> captured_amp = camera_prop(slm_phase)

    slm_phase: phase at the SLM plane, with dimensions [batch, 1, height, width]
    captured_amp: amplitude at the target plane, with dimensions [batch, 1, height, width]
    """

    def __init__(
        self,
        slm_device,
        cam_device,
        slm_show_time,
        slm_settle_time,
        channel=1,
        # roi_res=(1600, 880),
        # num_circles=(21, 12),
        roi_res=(640, 880),
        num_circles=(
            9,
            12,
        ),  # TODO (3, 3) makes dimension flip: width/height, and does not help the blob detector
        # laser_arduino=False,
        # com_port="COM3",
        # arduino_port_num=(6, 10, 11),
        # range_row=(200, 1000),
        # range_col=(300, 1700),
        range_row=(0, 768),  # TODO adapt to capture roi in real image
        range_col=(0, 1024),
        patterns_path="./citl/calibration",
        show_preview=False,
    ):
        super(PhysicalProp, self).__init__()

        # 1. Connect Camera
        # self.camera = CameraCapture()
        self.camera = camera.create_camera(cam_device)
        # self.camera.connect(0)  # specify the camera to use, 0 for main cam, 1 for the second cam

        # 2. Connect SLM
        # self.slm = SLMDisplay()
        # self.slm.connect()
        self.slm_settle_time = slm_settle_time
        self.slm = slm.create_slm(slm_device)
        self.slm.set_show_time(slm_show_time)

        # # 3. Connect to the Arduino that switches rgb color through the laser control box.
        # if laser_arduino:
        #     self.alc = ArduinoLaserControl(com_port, arduino_port_num)
        #     self.alc.switch_control_box(channel)
        # else:
        #     self.alc = None

        # 4. Calibrate hardwares using homography
        calib_ptrn_path = os.path.join(patterns_path, f'{("red", "green", "blue")[channel]}.png')
        space_btw_circs = [
            int(roi / (num_circs - 1)) for roi, num_circs in zip(roi_res, num_circles)
        ]

        self.calibrate(
            calib_ptrn_path,
            num_circles,
            space_btw_circs,
            range_row=range_row,
            range_col=range_col,
            show_preview=show_preview,
        )

    def calibrate(
        self,
        calibration_pattern_path,
        num_circles,
        space_btw_circs,
        range_row,
        range_col,
        show_preview=False,
        num_grab_images=10,
    ):
        """
        pre-calculate the homography between target plane and the camera captured plane

        :param calibration_pattern_path:
        :param num_circles:
        :param space_btw_circs: number of pixels between circles
        :param slm_settle_time:
        :param range_row:
        :param range_col:
        :param show_preview:
        :param num_grab_images:
        :return:
        """

        self.calibrator = Calibration(num_circles, space_btw_circs)

        # supposed to be a grid pattern image for calibration
        calib_phase_img = skimage.io.imread(calibration_pattern_path)

        calib_phase_img = np.mean(calib_phase_img[:, :, 0:3], axis=2)

        captured_img = self._capture_and_average_intensities(calib_phase_img, num_grab_images)

        # masking out dot pattern region for homography
        captured_img_masked = captured_img[
            range_row[0] : range_row[1], range_col[0] : range_col[1], ...
        ]

        calib_success = self.calibrator.calibrate(captured_img_masked, show_preview=show_preview)

        self.calibrator.start_row, self.calibrator.end_row = range_row
        self.calibrator.start_col, self.calibrator.end_col = range_col

        if calib_success:
            print("   - Calibration succeeded")
        else:
            # raise ValueError("   - Calibration failed") # TODO switch back to
            # raise error
            print("   - Calibration failed")

    def forward(self, slm_phase, num_grab_images=1):
        """
        this forward pass gets slm_phase to display and returns the amplitude image at the target plane.

        :param slm_phase:
        :param num_grab_images:
        :return: A pytorch tensor shape of (1, 1, H, W)
        """
        slm_phase_8bit = utils.phasemap_8bit(slm_phase, True)

        # display the pattern and capture linear intensity, after perspective transform
        captured_linear_np = self.capture_linear_intensity(
            slm_phase_8bit, num_grab_images=num_grab_images
        )

        # convert raw-16 linear intensity image into an amplitude tensor
        if len(captured_linear_np.shape) > 2:
            captured_linear = (
                torch.tensor(captured_linear_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            )
            captured_linear = captured_linear.to(slm_phase.device)
            captured_linear = torch.sum(captured_linear, dim=1, keepdim=True)
        else:
            captured_linear = (
                torch.tensor(captured_linear_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )
            captured_linear = captured_linear.to(slm_phase.device)

        # return amplitude
        return torch.sqrt(captured_linear)

    def capture_linear_intensity(self, slm_phase, num_grab_images):
        """
        TODO complete doc

        :param slm_phase:
        :param num_grab_images:
        :return:
        """

        captured_intensity_raw_avg = self._capture_and_average_intensities(
            slm_phase, num_grab_images
        )

        # crop ROI as calibrated
        captured_intensity_raw_cropped = captured_intensity_raw_avg[
            self.calibrator.start_row : self.calibrator.end_row,
            self.calibrator.start_col : self.calibrator.end_col,
            ...,
        ]
        # apply homography
        return self.calibrator(captured_intensity_raw_cropped)

    def _capture_and_average_intensities(self, phase_map, num_grab_images):
        self.slm.imshow(phase_map)
        time.sleep(self.slm_settle_time)
        captured_intensities = self.camera.acquire_multiple_images_and_resize_to_slm_shape(
            num_grab_images
        )
        return utils.burst_img_processor(captured_intensities)

    # def disconnect(self):
    #     self.camera.disconnect()
    #     self.slm.disconnect()
    #     if self.alc is not None:
    #         self.alc.turnOffAll()
