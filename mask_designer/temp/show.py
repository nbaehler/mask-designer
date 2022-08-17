import pickle
import sys

from os.path import dirname, abspath, join  # TODO everywhere??
import sys

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)


from slm_controller import slm
from mask_designer.experimental_setup import slm_device


def show(phase_map_path):
    import datetime

    s = slm.create(slm_device)

    with open(phase_map_path, "rb") as f:
        phase_map = pickle.load(f)

    print(datetime.datetime.now().time(), "Inside imshow")
    s.imshow(phase_map)  # TODO comment out
    print(datetime.datetime.now().time(), "End imshow")


if __name__ == "__main__":
    args = sys.argv[1:]
    show(args[0])
