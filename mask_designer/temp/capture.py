import pickle
import time

from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)


from mask_designer import camera
from mask_designer.experimental_setup import cam_device


def capture(exposure_time, num_grab_images, resize, slm_settle_time, captures_path):
    import datetime

    cam = camera.create(cam_device)
    cam.set_exposure_time(exposure_time)

    print(datetime.datetime.now().time(), "Start settle")
    time.sleep(slm_settle_time)
    print(datetime.datetime.now().time(), "End settle, start capture")

    if resize:
        captured_intensities = cam.acquire_multiple_images_and_resize_to_slm_shape(num_grab_images)
    else:
        captured_intensities = cam.acquire_multiple_images(num_grab_images)

    print(datetime.datetime.now().time(), "End capture")

    pickle.dump(captured_intensities, open(captures_path, "wb"))


if __name__ == "__main__":
    args = sys.argv[1:]
    capture(float(args[0]), int(args[1]), bool(args[2]), float(args[3]), args[4])
