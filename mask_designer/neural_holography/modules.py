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

import datetime
import os
import pickle
import time
from multiprocessing import Process
from pathlib import Path

import mask_designer.neural_holography.utils as utils
import numpy as np
import torch
import torch.nn as nn
from mask_designer.experimental_setup import slm_device
from mask_designer.neural_holography.algorithms import (
    gerchberg_saxton,
    stochastic_gradient_descent,
)
from mask_designer.neural_holography.calibration_module import Calibration
from mask_designer.neural_holography.prop_asm import prop_asm

# from mask_designer.transform_fields import neural_holography_lensless_to_lens # TODO Circular import
from mask_designer.utils import (
    angularize_phase_mask,
    extend_to_field,
    load_image,
    normalize_mask,
    quantize_phase_mask,
    round_phase_mask_to_uint8,
    save_image,
)
from PIL import Image
from slm_controller.hardware import SLMParam, slm_devices

# from mask_designer.simulate_prop import (  # TODO Circular import
# holoeye_fraunhofer,
# simulate_prop,
# )


class GS(nn.Module):
    """Classical Gerchberg-Saxton algorithm

    :param prop_distance: propagation dist between SLM and target, in meters
    :param wavelength: the wavelength of interest, in meters
    :param feature_size: the SLM pixel pitch, in meters
    :param num_iters: the number of iteration, default 500
    :param loss: loss function, default L2
    :param prop_model: chooses the propagation operator ('ASM': propagation_ASM,
        'model': calibrated model). Default 'ASM'.
    :param propagator: propagator instance (function / pytorch module)
    :param device: torch.device

    Functions as a pytorch module:

    >>> gs = GS(...)
    >>> final_phase = gs(target_amp, init_phase)

    :param target_amp: amplitude at the target plane, with dimensions [batch, 1, height, width]
    :param init_phase: initial guess of phase of phase-only slm
    :var final_phase: optimized phase-only representation at SLM plane, same dimensions
    """

    def __init__(
        self,
        prop_distance,
        wavelength,
        feature_size,
        num_iters,
        prop_model="ASM",
        propagator=prop_asm,
        writer=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(GS, self).__init__()

        # Setting parameters
        self.prop_distance = prop_distance
        self.wavelength = wavelength
        self.feature_size = feature_size
        self.precomputed_H_f = None
        self.precomputed_H_b = None
        self.prop_model = prop_model
        self.propagator = propagator
        self.num_iters = num_iters
        self.writer = writer
        self.dev = device

    def forward(self, target_amp, init_phase=None):
        # Pre-compute propagation kernel only once
        if self.precomputed_H_f is None and self.prop_model == "ASM":
            self.precomputed_H_f = self.propagator(
                torch.empty(*init_phase.shape, dtype=torch.complex64),
                self.feature_size,
                self.wavelength,
                self.prop_distance,
                return_H=True,
            )
            self.precomputed_H_f = self.precomputed_H_f.to(self.dev).detach()
            self.precomputed_H_f.requires_grad = False

        if self.precomputed_H_b is None and self.prop_model == "ASM":
            self.precomputed_H_b = self.propagator(
                torch.empty(*init_phase.shape, dtype=torch.complex64),
                self.feature_size,
                self.wavelength,
                -self.prop_distance,
                return_H=True,
            )
            self.precomputed_H_b = self.precomputed_H_b.to(self.dev).detach()
            self.precomputed_H_b.requires_grad = False

        # Run algorithm
        return gerchberg_saxton(
            init_phase,
            target_amp,
            self.num_iters,
            self.prop_distance,
            self.wavelength,
            self.feature_size,
            prop_model=self.prop_model,
            propagator=self.propagator,
            precomputed_H_f=self.precomputed_H_f,
            precomputed_H_b=self.precomputed_H_b,
        )


class SGD(nn.Module):
    """Proposed Stochastic Gradient Descent Algorithm using Auto-diff Function of PyTorch

    :param prop_distance: propagation dist between SLM and target, in meters
    :param wavelength: the wavelength of interest, in meters
    :param feature_size: the SLM pixel pitch, in meters
    :param num_iters: the number of iteration, default 500
    :param roi_res: region of interest to penalize the loss
    :param loss: loss function, default L2
    :param prop_model: chooses the propagation operator ('ASM': propagation_ASM,
        'model': calibrated model). Default 'ASM'.
    :param propagator: propagator instance (function / pytorch module)
    :param lr: learning rate for phase variables
    :param lr_s: learning rate for the learnable scale
    :param s0: initial scale
    :param writer: SummaryWrite instance for tensorboard
    :param device: torch.device

    Functions as a pytorch module:

    >>> sgd = SGD(...)
    >>> final_phase = sgd(target_amp, init_phase)

    :param target_amp: amplitude at the target plane, with dimensions [batch, 1, height, width]
    :param init_phase: initial guess of phase of phase-only slm
    :var final_phase: optimized phase-only representation at SLM plane, same dimensions
    """

    def __init__(
        self,
        prop_distance,
        wavelength,
        feature_size,
        num_iters,
        roi_res,
        prop_model="ASM",
        propagator=prop_asm,
        loss=nn.MSELoss(),
        lr=0.01,
        lr_s=0.003,
        s0=1.0,
        writer=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(SGD, self).__init__()

        # Setting parameters
        self.prop_distance = prop_distance
        self.wavelength = wavelength
        self.feature_size = feature_size
        self.roi_res = roi_res
        self.precomputed_H = None
        self.prop_model = prop_model
        self.propagator = propagator

        self.num_iters = num_iters
        self.lr = lr
        self.lr_s = lr_s
        self.init_scale = s0

        self.writer = writer
        self.dev = device
        self.loss = loss.to(device)

    def forward(self, target_amp, init_phase=None):
        # Pre-compute propagation kernel only once
        if self.precomputed_H is None and self.prop_model == "ASM":
            self.precomputed_H = self.propagator(
                torch.empty(*init_phase.shape, dtype=torch.complex64),
                self.feature_size,
                self.wavelength,
                self.prop_distance,
                return_H=True,
            )
            self.precomputed_H = self.precomputed_H.to(self.dev).detach()
            self.precomputed_H.requires_grad = False

        # Run algorithm
        return stochastic_gradient_descent(
            init_phase,
            target_amp,
            self.num_iters,
            self.prop_distance,
            self.wavelength,
            self.feature_size,
            roi_res=self.roi_res,
            prop_model=self.prop_model,
            propagator=self.propagator,
            loss=self.loss,
            lr=self.lr,
            lr_s=self.lr_s,
            s0=self.init_scale,
            writer=self.writer,
            precomputed_H=self.precomputed_H,
        )


class PropPhysical(nn.Module):
    """A module for physical propagation,
    forward pass displays gets phase mask as an input and display the mask on the physical setup,
    and capture the diffraction image at the target plane,
    and then return warped image using pre-calibrated homography from instantiation.

    :param channel:
    :param slm_settle_time:
    :param roi_res:
    :param num_circles:
    :param range_row:
    :param range_col:
    :param pattern_path:
    :param calibration_preview:

    Functions as a pytorch module:

    >>> prop_physical = PropPhysical(...)
    >>> captured_amp = prop_physical(slm_phase)

    :param slm_phase: phase at the SLM plane, with dimensions [batch, 1, height, width]
    :var captured_amp: amplitude at the target plane, with dimensions [batch, 1, height, width]
    """

    def __init__(
        self,
        slm,
        slm_settle_time,
        slm_show_time,
        cam,
        roi_res,
        prop_distance,
        wavelength,
        channel=1,
        num_circles=(9, 12),
        range_row=(0, 768),  # TODO adapt to capture roi in real image
        range_col=(0, 1024),
        pattern_path="./citl/calibration",
        show_preview=False,
    ):
        super(PropPhysical, self).__init__()

        # 1. Connect Camera
        self.camera = cam

        # 2. Connect SLM
        self.slm = slm
        self.slm_settle_time = slm_settle_time
        self.slm_show_time = slm_show_time

        # 3. Set parameters
        self.prop_distance = prop_distance
        self.wavelength = wavelength
        self.slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]
        self.pixel_pitch = slm_devices[slm_device][SLMParam.PIXEL_PITCH]

        # 3. Calibrate hardwares using homography
        calib_pattern_path = os.path.join(pattern_path, f'{("red", "green", "blue")[channel]}.png')
        space_btw_circs = [
            int(roi / (num_circs - 1)) for roi, num_circs in zip(roi_res, num_circles)
        ]

        self.calibrate(
            calib_pattern_path,
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

        blank_phase_mask = np.zeros(slm_devices[slm_device][SLMParam.SLM_SHAPE], dtype=np.uint8,)

        captured_blank = self._capture_and_average_intensities(
            num_grab_images, False, blank_phase_mask, True,
        )

        self.camera.set_correction(captured_blank)

        # supposed to be a grid pattern image for calibration
        calib_phase_mask = load_image(calibration_pattern_path)

        captured_img = self._capture_and_average_intensities(
            num_grab_images, True, calib_phase_mask, True,
        )

        self.slm.set_show_time(self.slm_show_time)

        # masking out dot pattern region for homography
        corrected_img_masked = captured_img[
            range_row[0] : range_row[1], range_col[0] : range_col[1], ...
        ]

        calib_success = self.calibrator.calibrate(corrected_img_masked, show_preview=show_preview)

        self.calibrator.start_row, self.calibrator.end_row = range_row
        self.calibrator.start_col, self.calibrator.end_col = range_col

        if calib_success:
            print("   - Calibration succeeded")
        else:
            # raise ValueError("   - Calibration failed") # TODO switch back
            print("   - Calibration failed")

    def forward(self, slm_phase, num_grab_images=1):
        """
        this forward pass gets slm_phase to display and returns the amplitude image at the target plane.

        :param slm_phase:
        :param num_grab_images:
        :return: A pytorch tensor shape of (1, 1, H, W)
        """
        slm_phase_8bit = utils.phasemap_8bit(slm_phase, False)

        # display the pattern and capture linear intensity, after perspective transform
        captured_linear_np = self.capture_linear_intensity(
            slm_phase_8bit, num_grab_images=num_grab_images
        )

        return torch.tensor(normalize_mask(captured_linear_np), dtype=torch.float32)[
            None, None, :, :
        ].to(slm_phase.device)

    def capture_linear_intensity(self, slm_phase, num_grab_images):
        """
        Capture images and average them. Then, crop the image to the ROI and apply perspective transform.

        :param slm_phase: phase at the SLM plane.
        :param num_grab_images: number of images to average.
        :return: Averaged captured image after perspective transform and crop.
        """

        captured_intensity_raw_avg = self._capture_and_average_intensities(
            num_grab_images, True, slm_phase
        )

        # crop ROI as calibrated
        captured_intensity_raw_cropped = captured_intensity_raw_avg[
            self.calibrator.start_row : self.calibrator.end_row,
            self.calibrator.start_col : self.calibrator.end_col,
            ...,
        ]
        # apply homography
        return self.calibrator(captured_intensity_raw_cropped)

    def _transform_phase_mask(self, phase_mask):
        field = extend_to_field(angularize_phase_mask(phase_mask))[None, None, :, :]

        from mask_designer.transform_fields import (
            neural_holography_lensless_to_lens,
        )  # TODO Circular import

        # Transform the results to the hardware setting using a lens
        field = neural_holography_lensless_to_lens(
            field, self.prop_distance, self.wavelength, self.slm_shape, self.pixel_pitch,
        )

        return quantize_phase_mask(field.angle())

    def _capture_subprocess(
        self, cam, slm_settle_time, num_grab_images, resize, captures_path,
    ):
        print(datetime.datetime.now().time(), "Start settle")
        time.sleep(slm_settle_time)
        print(datetime.datetime.now().time(), "End settle, start capture")

        if resize:
            captured_intensities = cam.acquire_multiple_images_and_resize_to_slm_shape(
                num_grab_images
            )
        else:
            captured_intensities = cam.acquire_multiple_images(num_grab_images)

        print(datetime.datetime.now().time(), "End capture")

        pickle.dump(captured_intensities, open(captures_path, "wb"))

    def _capture_and_average_intensities(
        self, num_grab_images, resize, phase_mask, calibration=False
    ):

        captures_path = Path("citl/captures.pkl")

        if captures_path.exists():
            captures_path.unlink()

        cam_process = Process(
            target=self._capture_subprocess,
            args=[self.camera, self.slm_settle_time, num_grab_images, resize, captures_path,],
        )

        cam_process.start()

        print(datetime.datetime.now().time(), "Start imshow")

        if not calibration:
            phase_mask = self._transform_phase_mask(phase_mask)

            field = extend_to_field(angularize_phase_mask(phase_mask))[None, None, :, :]

            from mask_designer.simulate_prop import (  # TODO Circular import
                holoeye_fraunhofer,
                simulate_prop,
            )

            propped_field = simulate_prop(field, holoeye_fraunhofer)

            name = str(datetime.datetime.now().time()).replace(":", "_").replace(".", "_")

            save_image(
                round_phase_mask_to_uint8(255 * normalize_mask(propped_field.abs())),
                f"citl/snapshots/sim_{name}.png",
            )

        self.slm.imshow(phase_mask)

        print(datetime.datetime.now().time(), "End imshow")

        if not captures_path.exists():
            if cam_process.is_alive():
                cam_process.terminate()

            raise ValueError("Image capturing process timed out!")

        with open(captures_path, "rb") as f:
            captures = pickle.load(f)

        if cam_process.is_alive():
            cam_process.terminate()

        captures_path.unlink()

        img = utils.burst_img_processor(captures)

        img_file = Image.fromarray(img)
        name = str(datetime.datetime.now().time()).replace(":", "_").replace(".", "_")
        img_file.save(f"citl/snapshots/phy_{name}.png")

        return img
