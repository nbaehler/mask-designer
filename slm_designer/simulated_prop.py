from matplotlib import pyplot as plt

from slm_designer.propagation import holoeye_fraunhofer


def simulated_prop(
    slm_field, propagation_method=holoeye_fraunhofer, *args
):  # TODO use everywhere?
    return propagation_method(slm_field, *args)[0, 0, :, :]


def plot_sim_result(propped_slm_field):
    # Plot
    _, ax = plt.subplots()
    ax.imshow(propped_slm_field.abs(), cmap="gray")
    plt.show()
