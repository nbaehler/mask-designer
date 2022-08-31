# mask-designer

Collection of techniques to design mask patterns, e.g. fixed apertures for
amplitude masks and phase retrieval for determining a phase mask under an
incoherent light source.

- [mask-designer](#mask-designer)
  - [Installation](#installation)
    - [Optional setup for Camera-In-The-Loop](#optional-setup-for-camera-in-the-loop)
  - [Phase retrieval algorithms](#phase-retrieval-algorithms)
  - [Camera](#camera)
  - [Experimental setup](#experimental-setup)
  - [Propagation](#propagation)
    - [Physical propagation](#physical-propagation)
  - [Setting common apertures](#setting-common-apertures)
  - [Example scripts](#example-scripts)
    - [CITL examples](#citl-examples)
    - [Camera examples](#camera-examples)
    - [Physical propagation examples](#physical-propagation-examples)
    - [Simulated propagation examples](#simulated-propagation-examples)
    - [Aperture examples](#aperture-examples)
  - [Adding a new camera](#adding-a-new-camera)

The main goal of the project is tackle the inverse problem called phase retrieval,
i.e. mask design for SLMs.
But it also allows to explore the forward problem by setting a phase
mask and then simply observing the output amplitude at the target plane.
Mainly though, it

consists of mask design techniques that were introduced in the [Neural
Holography](https://www.computationalimaging.org/publications/neuralholography/)
work by Prof. Gordon Wetzstein's group at Stanford, and adapted to be compliant
with the SLM of this project - the [Holoeye LC
2012](https://holoeye.com/lc-2012-spatial-light-modulator/). Further, another
part of the code does add support for cameras. Neural Holography, amongst
others, follows a **Camera-In-The-Loop** approach which involves a
camera taking pictures of the resulting interference patterns at the target
plane and then using this information to improve the designed mask iteratively.
Finally, utility functions are provided so that masks designed by Holoeye's
(closed-source) software or Neural Holography's code can be used
interchangeably for either setup (as Holoeye and Neural Holography assume
different physical setups).

If you wish to learn more about the evolution of our experimental setup, please
refer to the `documentation/DOCUMENTATION.md`.

<!-- TODO ReadTheDocs -->

Below is a schematic of how `mask-designer` would typically interact with other
components.

![Schematic representation of the interactions between different
components](docs/source/images/structure.svg)

The interactions marked with _CITL_ are only necessary for the CITL approach.
You can find animated gif-files showing those interactions in more details in
`documentation/DOCUMENTATION.md`.

## Installation

To install, simply run the following script:

```sh
./env_setup.sh
```

The script will:

1. Create a Python3 virtual environment called `mask_designer_env`.
2. Install Python dependencies in the virtual environment.
3. Install both [slm-controller](https://github.com/ebezzam/slm-controller) and
   [waveprop](https://github.com/ebezzam/waveprop) in setuptools “develop mode”
   from GitHub directly.

This project is using those two repositories to access physical SLMs after the
phase mask has been computed and to simulate the light propagation in the different
phase retrieval algorithms or propagation simulations. Note that those are still
in development too.

If you plan to use this code base more in depth you can install additional
dependencies intended for developing while the virtual environment is activated.

```sh
source mask_designer_env/bin/activate
# pip install -e .[dev] #TODO does not work, dev not found
pip install click black pytest tensorboard torch_tb_profiler
```

### Optional setup for Camera-In-The-Loop

Camera-In-The-Loop (CITL) obviously requires a camera. For this project we made
use of this camera [Thorlabs
DCC3260M](https://www.thorlabs.com/thorproduct.cfm?partnumber=DCC3260M) (which is unfortunately obsolete). If you would like to use
the same camera for the CITL technique introduced by Neural Holography, you will
have to install IDS software as well. First, install [IDS Software
Suite](https://en.ids-imaging.com/download-details/AB00695.html), simply follow
the installation instructions. Next, you need [IDS
Peak](https://en.ids-imaging.com/download-details/AB00695.html) which can be
found under the same link.
This is a software package containing some GUI applications to interact with the
camera but, most importantly, contains two wheel-packages that allow to install
the python API for the aforementioned camera. First, just follow the install
instructions but then make sure to pick the `Custom` installation option to add an
important feature. Once in this selection prompt,
check the box additionally installing `Support of uEye cameras` and continue.
After the installation is completed go to the installation directory. Note
that you should have activated the virtual environment created earlier from now
on (`source mask_designer_env/bin/activate`). Next, go to `ids_peak/generic_sdk/api/binding/python/wheel` and
install the appropriate wheel like so, for example:

```sh
pip install x86_64/ids_peak-1.4.1.0-cp39-cp39-win_amd64.whl
```

Secondly, again from the IDS installation
directory, go to `ids_peak/generic_sdk/ipl/binding/python/wheel`. Similar to
before, install the correct version of the wheel for your setup, for example:

```sh
pip install x86_64/ids_peak_ipl-1.3.2.7-cp39-cp39-win_amd64.whl
```

Now, you should be good to go to use all the features implemented in this
project.

## Phase retrieval algorithms

The authors of [Neural Holography](https://www.computationalimaging.org/publications/neuralholography/)
provide implementations to different phase retrieval approaches. Here is a list
of methods that were modified in order to be compatible with the hardware and
software components as shown in the above schematic:

- Gerchberg-Saxton (GS)
- Stochastic Gradient Descent (SGD)
- Double Phase Amplitude Coding (DPAC) <!-- TODO Remove if it's not working -->
- Camera-In-The-Loop (CITL)

GS, SGD and DPAC are all implemented inside `mask_designer/neural_holography/algorithms.py`
and PyTorch modules that go along with them are provided in
`mask_designer/neural_holography/module.py`. CITL on the other hand is located in a separate
script `mask_designer/neural_holography/train_model.py`. Note that you do
generally not need to interact with the Neural Holography code directly. A
wrapper for it is provided at `mask_designer/wrapper.py` which does simply import
code from Neural Holography so that you do not need to go look for it in their
code and also contains some interfacing methods to run the different phase
retrieval algorithms. We'd like to remind that
this code was released under the license provided in `LICENSE` and we do not
claim any credit for it. Usage examples of all
those features will be presented in the
subsequent [Example scripts](#example-scripts) section.

<!-- TODO add our own license -->

## Camera

As mentioned earlier, cameras play a crucial role in the CITL-approach. Hence, an
interface for such devices is needed. For now, the project only supports one
real camera, the [Thorlabs
DCC3260M](https://www.thorlabs.com/thorproduct.cfm?partnumber=DCC3260M) and a
dummy camera that simply "takes" black snapshots. The later can be useful during
development. In the future this list is going to be extended (for example with
the [Raspberry Pi HQ Camera](https://www.adafruit.com/product/4561)), but here
is its current state.

Supported cameras:

- Dummy camera (artificial, returns synthetic pitch black images)
- [Thorlabs DCC3260M](https://www.thorlabs.com/thorproduct.cfm?partnumber=DCC3260M)

## Experimental setup

The experimental setup is an incremental improvement of an initial setup proposed
by Holoeye in the manual that came with their their [LC 2012
SLM](https://holoeye.com/lc-2012-spatial-light-modulator/). For more information
on how we converged to the setup below, please refer to `documentation/DOCUMENTATION.md`.

![Experimental setup](docs/source/images/setup.svg)

Further, the `mask_designer/experimental_setup.py` allows one to set:

- which camera and SLM are used,
- how long masks are shown on the SLM,
- what wavelength the laser is operating at and, finally,
- the propagation distance (distance form the SLM to the camera sensor).

Those parameters are then used in the remainder of the code base.

<!-- TODO might not be only linked to lenses, ASM vs Fraunhofer -->

Holoeye provides a graphical software called [SLM Pattern
Generator](https://customers.holoeye.com/slm-pattern-generator-v5-1-1-windows/)
that has built-in functionality for performing phase retrieval for a given
target amplitude. One such example can be found in `images/holoeye_phase_mask`
and its corresponding amplitude at the target plane under `images/target_amplitude`.
This software assumes an experimental setup that uses a convex lens in between the SLM and
the target plane.

Neural Holography on the other hand, uses a different setting
where no lens is placed between the SLM and the target plane, i.e. a lensless
setting. Those differences impact the resulting phase masks of the mask design
algorithm. The methods in `mask_designer/transform_fields.py` allow
transforming phase maps, or fields, back and forth between both experimental
setups. Note that Neural Holography encodes
phase maps, images etc. as 4D PyTorch Tensors where the dimensions are [image,
channel, height, width]. But again, the wrapper `mask_designer/wrapper.py` does
provide interfacing methods for the different algorithms that handle all those
complications for you and you are not required to dig any deeper than that.

## Propagation

<!-- TODO might not be only linked to lenses, ASM vs Fraunhofer -->

This section will briefly discuss the propagation of a phase mask to the target
plane. More precisely, propagation simulation is a crucial element in most of the
mask designing algorithms. Although we cannot be absolutely certain due to the code being closed-source, we
have good reason to believe that Holoeye's SLM Pattern Generator uses
[Fraunhofer](https://en.wikipedia.org/wiki/Fraunhofer_diffraction_equation), as
we have identified a single Fourier Transform between the SLM and target plane
when playing around with their masks. Neural Holography on the other hand,
uses the [Angular spectrum
method](https://en.wikipedia.org/wiki/Angular_spectrum_method) (ASM). Currently,
we make use of the ASM implementation by Neural Holography. However we plan to
replace this implementation with the
[`waveprop`](https://github.com/ebezzam/waveprop) library, which provides
support for Fraunhofer, ASM, and other propagation techniques.

<!-- TODO replace prop with waveprop -->

### Physical propagation

Physical propagation refers to the process of physically displaying a phase map
on a SLM and then observing the resulting images at the target plane. That's where the
[slm-controller](https://github.com/ebezzam/slm-controller) comes in handy to
communicate with the physical SLMs, and the camera in order to measure the
response at the target plane.

<!-- TODO must change if we decide to always plot the masks -->

Note that this software package simply plots
the phase mask whenever something goes wrong with showing it on the physical
device so that you can still get an idea of the resulting phase maps.

Usage examples will be presented in the
subsequent [Example scripts](#example-scripts) section.

## Setting common apertures

The `mask_designer/aperture.py` provides
an easy way to set different apertures: rectangle, square, line, and circle.
These apertures can be programmed to real SLM devices. Usage example will be
presented in the subsequent [Example scripts](#example-scripts) section.

## Example scripts

<!-- TODO adapt example paths -->

In `examples` are various example scripts that showcase the main functionality
of this repository.

First, activate the virtual environment:

```sh
source mask_designer_env/bin/activate
```

You can exit the virtual environment by running `deactivate`.

### CITL examples

<!-- TODO adapt here -->

This section does show how CITL can be used. Note though that this is still
very much in development, so not bug free. More work will be needed here.

This script calls via the `mask_designer/wrapper.py` Neural Holography code that
evaluates the resulting amplitudes using different measures.

```sh
$ python examples/citl_eval.py --help
Usage: citl_eval.py [OPTIONS]

Options:
  --channel INTEGER        red:0, green:1, blue:2, rgb:3
  --prop_model TEXT        Type of propagation model for reconstruction: ASM /
                           MODEL / CAMERA
  --root_path TEXT         Directory where test phases are being stored.
  --prop_model_dir TEXT    Directory for the CITL-calibrated wave propagation
                           models
  --calibration_path TEXT  Directory where calibration phases are being
                           stored.
  --help                   Show this message and exit.
```

This code is very similar to the `mask_designer/neural_holography/eval.py` code
and needs further adaptions to simply output the phase mask without doing evaluation.

```sh
$ python examples/citl_predict.py --help
Usage: citl_predict.py [OPTIONS]

Options:
  --channel INTEGER        red:0, green:1, blue:2, rgb:3
  --prop_model TEXT        Type of propagation model for reconstruction: ASM /
                           MODEL / CAMERA
  --root_path TEXT         Directory where test phases are being stored.
  --prop_model_dir TEXT    Directory for the CITL-calibrated wave propagation
                           models
  --calibration_path TEXT  Directory where calibration phases are being
                           stored.
  --help                   Show this message and exit.
```

Finally, this script starts a CITL training session. The training process is
functional but more work to ensure that actual progress is made during training
is still needed.

```sh
$ python examples/citl_train.py --help
Usage: citl_train.py [OPTIONS]

Options:
  --channel INTEGER        red:0, green:1, blue:2, rgb:3
  --pretrained_path TEXT   Path of pretrained checkpoints as a starting point
  --model_path TEXT        Directory for saving out checkpoints
  --phase_path TEXT        Directory for precalculated phases
  --calibration_path TEXT  Directory where calibration phases are being stored
  --train_data_path TEXT   Directory where train data is stored.
  --lr_model FLOAT         Learning rate for model parameters
  --lr_phase FLOAT         Learning rate for phase
  --num_epochs INTEGER     Number of epochs
  --batch_size INTEGER     Size of minibatch
  --step_lr STR2BOOL       Use of lr scheduler
  --experiment TEXT        Name of the experiment
  --help                   Show this message and exit.
```

### Camera examples

This file illustrates how a camera, here the `IDSCamera`, is instantiated and
used to take a single image. The resulting image is then plotted to the screen.

```sh
python examples/ids_image_capture.py
```

### Physical propagation examples

This section contains example scripts, for sending both phase masks created using the
Holoeye software and phase masks generated using Neural Holography methods to
real SLM devices. Those mask are then propagated through our experimental
setup. Note that the
methods in `mask_designer/wrapper.py` are extensively used here to compute the
phase maps. We are going through them one by one now.

This script simply sets some parameters like wavelength etc., then loads a
target image (Holoeye logo) and runs the DPAC method. The resulting phase
mask is finally submitted to a real SLM.

<!-- TODO might just remove DPAC -->

```sh
$ python examples/physical_prop_dpac.py --help
Usage: physical_prop_dpac.py [OPTIONS]

Options:
  --slm_show_time FLOAT  Time to show the mask on the SLM.
  --help             Show this message and exit.
```

The next script does basically the same just using the GS method.
Additionally, it needs a random input phase mask that is going to be optimized
and you can set the number of iterations.

```sh
$ python examples/physical_prop_gs.py --help
Usage: physical_prop_gs.py [OPTIONS]

Options:
  --iterations INTEGER  Number of iterations to run.
  --slm_show_time FLOAT     Time to show the mask on the SLM.
  --help                Show this message and exit.
```

Unlike before, this script does not perform any computation. Instead it
only loads a precomputed phase map
generated using Holoeye's [SLM Pattern
Generator](https://customers.holoeye.com/slm-pattern-generator-v5-1-1-windows/)
software (again, for the Holoeye logo).

```sh
$ python examples/physical_prop_holoeye.py --help
Usage: physical_prop_holoeye.py [OPTIONS]

Options:
  --slm_show_time FLOAT  Time to show the mask on the SLM.
  --help             Show this message and exit.
```

Similar to GS, for SGD you can also specify the number of iterations you want to
perform and a random initial state of the phase mask is required.

```sh
$ python examples/physical_prop_sgd.py --help
Usage: physical_prop_sgd.py [OPTIONS]

Options:
  --iterations INTEGER  Number of iterations to run.
  --slm_show_time FLOAT     Time to show the mask on the SLM.
  --help                Show this message and exit.
```

### Simulated propagation examples

Same as above, different versions of the simulated propagation do exist, one for
a precomputed Holoeye phase map, another 3 for the phase masks computed with Neural
Holography methods and finally one that test a whole bunch of methods
implemented in waveprop. For the former four, as a sanity check, each phase map
is transformed into both lens and lensless setup and then its propagation is
simulated in the respective setting. The
resulting amplitudes must be the same. The script using waveprop methods
simply propagates the same precomputed phase mask as in the Holoeye script in a
variety of different ways.

The only difference to the physical propagation scripts is that here the
propagation is simulated and the results plotted to the screen.

```sh
python examples/simulated_prop_dpac.py
```

```sh
$ python examples/simulated_prop_gs.py --help
Usage: simulated_prop_gs.py [OPTIONS]

Options:
  --iterations INTEGER  Number of iterations to run.
  --help                Show this message and exit.
```

```sh
python examples/simulated_prop_holoeye.py
```

```sh
$ python examples/simulated_prop_sgd.py --help
Usage: simulated_prop_sgd.py [OPTIONS]

Options:
  --iterations INTEGER  Number of iterations to run.
  --help                Show this message and exit.
```

The next script on the other hand is more for development and checking different
methods import from waveprop. Not all methods are integrated correctly, more
work is also needed here.

```sh
python examples/simulated_prop_waveprop.py
```

### Aperture examples

To set a defined aperture shape, check out the following script:

```sh
$ python examples/set_aperture.py --help
Usage: set_aperture.py [OPTIONS]

  Set aperture on a physical device.

Options:
  --shape [rect|square|line|circ]
                                  Shape of aperture.
  --n_cells INTEGER               Side length for 'square', length for 'line',
                                  radius for 'circ'. To set shape for 'rect',
                                  use`rect_shape`.
  --rect_shape INTEGER...         Shape for 'rect' in number of cells; `shape`
                                  must be set to 'rect'.
  --center INTEGER...             Coordinate for center.
  --vertical                      Whether line should be vertical (True) or
                                  horizontal (False).
  --device [rgb|binary|nokia|holoeye]
                                  Which device to program with aperture.
  --help                          Show this message and exit.
```

For example, to create a circle aperture on the monochrome device with a radius of 20 cells:

```sh
python examples/set_aperture.py --device binary --shape circ --n_cells 20
```

For a square aperture on the RGB device with a side length of 2 cells:

```sh
python examples/set_aperture.py --device rgb --shape square --n_cells 2
```

You can preview an aperture with the following script. Note that it should be run on a machine with
plotting capabilities, i.e. with `matplotlib`.

```sh
$ python examples/plot_aperture.py --help
Usage: plot_aperture.py [OPTIONS]

  Plot SLM aperture.

Options:
  --shape [rect|square|line|circ]
                                  Shape of aperture.
  --n_cells INTEGER               Side length for 'square', length for 'line',
                                  radius for 'circ'. To set shape for 'rect',
                                  use`rect_shape`.
  --rect_shape INTEGER...         Shape for 'rect' in number of cells; `shape`
                                  must be set to 'rect'.
  --vertical                      Whether line should be vertical (True) or
                                  horizontal (False).
  --show_tick_labels              Whether or not to show cell values along
                                  axes.
  --pixel_pitch FLOAT...          Shape of cell in meters (height, width).
  --slm_shape INTEGER...          Dimension of SLM in number of cells (height,
                                  width).
  --device [rgb|binary|nokia|holoeye]
                                  Which device to program with aperture.
  --help                          Show this message and exit.
```

For example, to plot a square aperture on the RGB device with a side length of 2 cells:

```sh
python examples/plot_aperture.py --shape square --n_cells 2 --device rgb
```

## Adding a new camera

In order to add support for a new camera, a few steps need to be taken. These are
done to avoid hard-coded values, but rather have global variables/definitions
that are accessible throughout the whole code base.

1. Add camera configuration in `mask_designer/hardware.py:cam_devices`.
2. Define a new class in `mask_designer/camera.py` for interfacing with the new
   camera (set parameters, take images, etc.).
3. Add to factory method `create` in `mask_designer/camera.py` for a
   conveniently one-liner to instantiate an object of the new camera.
