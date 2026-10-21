import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MaxNLocator, EngFormatter


plt.rcParams.update({'font.size': 15})
plt.rcParams.update({'figure.autolayout': True})

def entries(x_values, x, L,  folder_name=None, file_prefix=None, log=False, show=True):
    """Plots the entries vs the iteration, saves the plot in folder_name if provided

    Parameters:
        x_values (list)     -       list of x values (list of np.arrays)
        x (np.array)        -       ground truth (to get the non-zero coordinates)
        L (integer)         -       depth of the network
        folder_name         -       Path where to store the plot
        file_prefix         -       string prefix for the name of the plot
        log (boolean)       -       if the x-axis should be logarithmic
        show (boolean)      -       if the plot should be shown
    """
    T = len(x_values)
    fig, ax = plt.subplots()

    temp = np.array(x_values)
    N = x_values[0].shape[0]

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Colours for the non-sparse entries
    # Standard labeling
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Entries')
    ax.grid(True)
    if log:
        ax.set_xscale('log')
        file_prefix += '_log'

    # Plotting the entries corresponding to the zeros of the sparse ground truth
    for t in np.setdiff1d(np.arange(N), np.nonzero(x)[0]):
        ax.plot(range(T), temp[:,t]**L, color='grey')
    i= 0 #Count index for the colours

    # Plotting the entries corresponding to the non-zero entries of the sparse ground truth
    for t in np.nonzero(x)[0]:
        ax.plot(range(T), temp[:,t]**L, color=colors[i])
        ax.plot([T+1], [x[t]], marker="*", markersize=3, color=colors[i])
        i=i+1

    # Automatically choose a readable number and spacing of ticks
    ax.xaxis.set_major_locator(
        MaxNLocator(
            nbins="auto",
            steps=[1, 2, 2.5, 5, 10],
            min_n_ticks=3,
            integer=True
        )
    )

    # Automatically use k, M, etc.
    ax.xaxis.set_major_formatter(EngFormatter(sep=""))

    fig.tight_layout()

    # Saving and plotting routine
    if folder_name is not None:
        plt.savefig(os.path.join(folder_name, 'Entries'+file_prefix+'.png'))
    if show:
        plt.show()
    plt.close()

def Loss_vs_Bregman_vs_H(loss=None, Bregman=None, H=None, H_v=None, log_x=False, log_y=True, folder_name=None, file_prefix=None, show=True):
    """Plots the Loss, the Bregman Divergence and the sum of error terms (cumulative) (or a subset of those) against the iterates.

    Parameters:
        loss            -       list of scalars (L(beta_t))
        Bregman         -       list of scalars (D_h(\beta^*, beta_t))
        H               -       list of scalars (sum_s^t H(eta_s, \beta_s, \nabla \L(\beta_s)) (if H_v is provided this is for u)
        H_v             -       list of scalars for the general model
        log_x           -       boolean, if the x-axis should be logarithmic
        log_y           -       boolean, if the y-axis should be logarithmic
        folder_name     -       Path where to store the plot
        file_prefix     -       string prefix for the name of the plot
        show            -       (boolean), if the plot should be shown
    """
    T = np.inf

    # Getting the number of iterates
    if loss is not None:
        T = len(loss)
    if Bregman is not None and len(Bregman)<T:
        T = len(Bregman)
    if H is not None and len(H)<T:
        T = len(H)

    fig, ax1 = plt.subplots()
    file_name = ''

    if loss is not None:
        ax1.plot(range(T), loss[:T], label=r'$L(\beta_t)$')
        file_name += 'Loss'
    if Bregman is not None:
        ax1.plot(range(T), Bregman[:T], label=r'$D_h(\beta_t, \beta^*)$')
        file_name += '_Bregman'
    ax1.set_xlabel('Iteration')

    color = 'tab:green'
    if H_v is not None:
        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
        ax2.plot(range(T), H[:T], label='\|H_u\|_1 cumulative', color=color)
        ax2.plot(range(T), H_v[:T], label='\|H_v\|_1 cumulative', color='green')
        ax2.tick_params(axis='y', labelcolor=color)
        file_name += '_H'
        ax2.legend()

    elif H is not None:
        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
        ax2.set_ylabel(r'$\|H(\eta_t, \beta_t, \nabla L(\beta_t))\|_1$ cumulative', color=color)
        ax2.plot(range(T), H[:T], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        file_name += '_H'

    if log_x and folder_name is not None:
        file_prefix += '_logx'
        ax1.xscale('log')
    if log_y and folder_name is not None:
        file_prefix += '_logy'
        ax1.set_yscale('log')
    ax1.grid()
    ax1.legend()

    # Automatically choose a readable number and spacing of ticks
    ax1.xaxis.set_major_locator(
        MaxNLocator(
            nbins="auto",
            steps=[1, 2, 2.5, 5, 10],
            min_n_ticks=3,
            integer=True
        )
    )

    # Automatically use k, M, etc.
    ax1.xaxis.set_major_formatter(EngFormatter(sep=""))

    fig.tight_layout()

    # Save everything
    if folder_name is not None:
        plt.savefig(os.path.join(folder_name, file_name + file_prefix+'.png'))
    if show:
        plt.show()
    plt.close()



def loss_compare_GD_SGD(loss_GD, loss_SGD, y_scale=False, folder_name=None, show=True):
    """Comparing Gradient Descent loss with SGD loss"""
    T = len(loss_GD)
    fig, ax = plt.subplots()
    ax.plot(range(T), loss_SGD, label='batch loss')
    ax.plot(range(T), loss_GD, label='full loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    file_suffix = ''
    if y_scale:
        plt.yscale('log')
        file_suffix += '_y_scale'

        # Automatically choose a readable number and spacing of ticks
        ax.xaxis.set_major_locator(
            MaxNLocator(
                nbins="auto",
                steps=[1, 2, 2.5, 5, 10],
                min_n_ticks=3,
                integer=True
            )
        )

        # Automatically use k, M, etc.
        ax.xaxis.set_major_formatter(EngFormatter(sep=""))

        fig.tight_layout()

    if folder_name is not None:
        plt.savefig(os.path.join(folder_name, 'GD_vs_SGD'+file_suffix+'.png'))
    if show:
        plt.show()

def eta_values(eta_values, folder_name=None, y_scale=True, x_scale=False, show=True):
    """ Plotting the eta_values chosen by the adaptive step-size strategy"""
    T = len(eta_values)
    prefix = ''
    plt.plot(range(T), eta_values)
    plt.xlabel('Iteration')
    plt.ylabel(r'$\eta$')
    plt.grid()
    if y_scale:
        plt.yscale('log')
        prefix = 'y_log_'
    if x_scale:
        plt.xscale('log')
        prefix = 'x_log_'
    if folder_name is not None:
        plt.savefig(os.path.join(folder_name, prefix+'eta.png'))
    if show:
        plt.show()
    plt.close()


def entries_compare_GD_SGD(entries_GD, entries_SGD, x, L, log=False, folder_name=None, show=True):
    """Comparing the entries obtained by Gradient Descent and Stochastic Gradient Descent"""
    T = len(entries_GD)
    S = len(entries_SGD)
    temp1 = np.array(entries_GD)
    temp2 = np.array(entries_SGD)
    N = x.shape[0]

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Colours we are going to use
    # Standard labeling
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Entries')
    if log:
        plt.xscale('log')

    # Plotting the entries corresponding to the zeros of the sparse ground truth
    #for t in np.setdiff1d(np.arange(N), np.nonzero(x)[0]):
    #    plt.plot(range(T), temp1[:,t]**L, color='grey')
    #    plt.plot(range(S), temp2[:,t]**L, color='grey')
    i= 0 #Count index for the colours

    # Plotting the entries corresponding to the non-zero entries of the sparse ground truth
    for t in np.nonzero(x)[0]:
        plt.plot(range(S), temp2[:,t]**L, label='SGD' , color=colors[1])
        plt.plot(range(T), temp1[:, t] ** L, label='GD', color=colors[0])
        plt.plot([T+1], [x[t]], marker="*", markersize=3, color=colors[i])
        i=i+1
    plt.legend()
    # Saving and plotting routine
    if folder_name is not None:
        plt.savefig(os.path.join(folder_name, '_entries.png'))
    if show:
        plt.show()
    plt.close()


def with_respect_to_step_size(eta_list, plot_values, y_label, name, x_label=None, y_lim=False, x_scale='linear', folder_name=None, show=True):
    """Plotting some data (plot_values) against the eta_list
    """
    T = len(eta_list)
    plt.grid()
    if x_label is None:
        plt.xlabel(r"Stepsize $\eta$")
    else:
        plt.xlabel(x_label)
    plt.ylabel(y_label)

    eta_array = np.asarray(eta_list)
    values_array = np.asarray(plot_values)
    order = np.argsort(eta_array)
    eta_sorted = eta_array[order]
    values_sorted = values_array[order]
    # For the general model the values will be for u, v separately
    if(len(values_array.shape)) == 2:
        plt.plot(eta_sorted, values_sorted[:,0], label='u')
        plt.plot(eta_sorted, values_sorted[:,1], label='v')
        plt.xscale('linear')
        plt.scatter(eta_sorted, values_sorted[:,0])
        plt.scatter(eta_sorted, values_sorted[:,1])
        plt.legend()
    else:
        plt.plot(eta_sorted, values_sorted)
        plt.xscale(x_scale)
        plt.scatter(eta_sorted, values_sorted)
    if y_lim:
        plt.ylim(0, float(y_lim))
    if folder_name is not None:
        plt.savefig(os.path.join(folder_name, name+'.png'))
    if show:
        plt.show()
    plt.close()

def losses_convergence_speed(eta_list, loss_list, name, folder_name=None, show=True):
    """Plotting the speed of convergence. This is done by plotting the loss decrease for several etas"""
    eta_array = np.asarray(eta_list)

    order = np.argsort(eta_array)

    for i in range(len(eta_list)):
        j = order[i]
        T = len(loss_list[j])
        plt.plot(range(T), loss_list[j], label=eta_list[j])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.xscale('log')
    if folder_name is not None:
        plt.savefig(os.path.join(folder_name, name+'.png'))
    if show:
        plt.show()
    plt.close()
