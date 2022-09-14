Example scripts
---------------


.. TODO adapt example paths



In ``examples`` are various example scripts that showcase the main functionality
of this repository.

First, activate the virtual environment:

.. code-block:: sh

   source mask_designer_env/bin/activate

You can exit the virtual environment by running ``deactivate``.

CITL examples
^^^^^^^^^^^^^


.. TODO adapt here



This section does show how CITL can be used. Note though that this is still
very much in development, so not bug free. More work will be needed here.

This script calls via the ``mask_designer/wrapper.py`` Neural Holography code that
evaluates the resulting amplitudes using different measures.

.. code-block:: sh

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

This code is very similar to the ``mask_designer/neural_holography/eval.py`` code
and needs further adaptions to simply output the phase mask without doing evaluation.

.. code-block:: sh

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

Finally, this script starts a CITL training session. The training process is
functional but more work to ensure that actual progress is made during training
is still needed.

.. code-block:: sh

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

Camera examples
^^^^^^^^^^^^^^^

This file illustrates how a camera, here the ``IDSCamera``, is instantiated and
used to take a single image. The resulting image is then plotted to the screen.

.. code-block:: sh

   python examples/ids_image_capture.py

Physical propagation examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section contains example scripts, for sending both phase masks created using the
Holoeye software and phase masks generated using Neural Holography methods to
real SLM devices. Those mask are then propagated through our experimental
setup. Note that the
methods in ``mask_designer/wrapper.py`` are extensively used here to compute the
phase maps. We are going through them one by one now.

This script simply sets some parameters like wavelength etc., then loads a
target image (Holoeye logo) and runs the GS method which needs a random input
phase mask that is going to be optimized
and you can set the number of iterations. The resulting phase
mask is finally submitted to a real SLM.

.. code-block:: sh

   $ python examples/physical_prop_gs.py --help
   Usage: physical_prop_gs.py [OPTIONS]

   Options:
     --iterations INTEGER  Number of iterations to run.
     --help                Show this message and exit.

Unlike before, this script does not perform any computation. Instead it
only loads a precomputed phase map
generated using Holoeye's `SLM Pattern
Generator <https://customers.holoeye.com/slm-pattern-generator-v5-1-1-windows/>`_
software (again, for the Holoeye logo).

.. code-block:: sh

   $ python examples/physical_prop_holoeye.py --help
   Usage: physical_prop_holoeye.py [OPTIONS]

   Options:
     --help             Show this message and exit.

Similar to GS, for SGD you can also specify the number of iterations you want to
perform and a random initial state of the phase mask is required.

.. code-block:: sh

   $ python examples/physical_prop_sgd.py --help
   Usage: physical_prop_sgd.py [OPTIONS]

   Options:
     --iterations INTEGER  Number of iterations to run.
     --help                Show this message and exit.

Simulated propagation examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

.. code-block:: sh

   $ python examples/simulated_prop_gs.py --help
   Usage: simulated_prop_gs.py [OPTIONS]

   Options:
     --iterations INTEGER  Number of iterations to run.
     --help                Show this message and exit.

.. code-block:: sh

   python examples/simulated_prop_holoeye.py

.. code-block:: sh

   $ python examples/simulated_prop_sgd.py --help
   Usage: simulated_prop_sgd.py [OPTIONS]

   Options:
     --iterations INTEGER  Number of iterations to run.
     --help                Show this message and exit.

The next script on the other hand is more for development and checking different
methods import from waveprop. Not all methods are integrated correctly, more
work is also needed here.

.. code-block:: sh

   python examples/simulated_prop_waveprop.py

Aperture examples
^^^^^^^^^^^^^^^^^

To set a defined aperture shape, check out the following script:

.. code-block:: sh

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

For example, to create a circle aperture on the monochrome device with a radius of 20 cells:

.. code-block:: sh

   python examples/set_aperture.py --device binary --shape circ --n_cells 20

For a square aperture on the RGB device with a side length of 2 cells:

.. code-block:: sh

   python examples/set_aperture.py --device rgb --shape square --n_cells 2

You can preview an aperture with the following script. Note that it should be run on a machine with
plotting capabilities, i.e. with ``matplotlib``.

.. code-block:: sh

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

For example, to plot a square aperture on the RGB device with a side length of 2 cells:

.. code-block:: sh

   python examples/plot_aperture.py --shape square --n_cells 2 --device rgb