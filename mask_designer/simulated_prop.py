from mask_designer.propagation import holoeye_fraunhofer
import matplotlib.pyplot as plt
from mask_designer.utils import normalize_mask


default_prop_method = holoeye_fraunhofer


def simulated_prop(field, propagation_method=default_prop_method, *args):
    return propagation_method(field, *args)[0, 0, :, :]


# def plot_sim_result(propped_field):  # TODO unused
#     # Plot
#     _, ax = plt.subplots()
#     ax.imshow(propped_field.abs(), cmap="gray")
#     plt.show()


def show_fields(field, propped_field, title):
    """
    Plotting utility function.

    Parameters
    ----------
    field : torch.Tensor
        The field before propagation
    propped_field : torch.Tensor
        The field after propagation
    title : String
        The title of the plot
    """
    fig = plt.figure()
    fig.suptitle(title)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.title.set_text("Phase on SLM")
    ax2.title.set_text("Amplitude on SLM")
    ax3.title.set_text("Phase after propagation to screen")
    ax4.title.set_text("Amplitude after propagation to screen")
    ax1.imshow(normalize_mask(field.angle()), cmap="gray")  # TODO normalize?
    ax2.imshow(normalize_mask(field.abs()), cmap="gray")
    ax3.imshow(normalize_mask(propped_field.angle()), cmap="gray")
    ax4.imshow(normalize_mask(propped_field.abs()), cmap="gray")
    plt.show()
