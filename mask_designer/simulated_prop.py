from mask_designer.propagation import holoeye_fraunhofer


default_prop_method = holoeye_fraunhofer


def simulated_prop(field, propagation_method=default_prop_method, *args):
    return propagation_method(field, *args)[0, 0, :, :]


# def plot_sim_result(propped_field):  # TODO unused
#     # Plot
#     _, ax = plt.subplots()
#     ax.imshow(propped_field.abs(), cmap="gray")
#     plt.show()
