# from matplotlib import pyplot as plt

from slm_designer.propagation import holoeye_fraunhofer

default_prop_method = holoeye_fraunhofer


def simulated_prop(phase_map, propagation_method=default_prop_method, *args):
    return propagation_method(phase_map, *args)[0, 0, :, :]


# def plot_sim_result(propped_phase_map):  # TODO unused
#     # Plot
#     _, ax = plt.subplots()
#     ax.imshow(propped_phase_map.abs(), cmap="gray")
#     plt.show()
