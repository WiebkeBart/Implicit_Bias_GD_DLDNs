import os

import json
from asyncio import tasks
from pathlib import Path
import numpy as np

from . import help_fcts as hf
from . import plots
from . import network_fcts as nf
from .save_routines import minimizer_values
import argparse

file_path = os.getcwd() #set the directory we are in


def read_config_file(dir:str) -> dict:
    """Function to return the config file for a specific directory."""

    folder = os.path.join(file_path, 'data/', dir)
    folder = Path(folder)
    with open(folder/ "config.json") as file:
        d = json.load(file)
        file.close()
    return d


def read_npz(dir:str) -> np.lib.npyio.NpzFile:
    """Function to return the results.npz file for a specific directory."""

    folder = os.path.join(file_path, 'data/', dir)
    folder = Path(folder)
    if os.path.isdir(folder)==False:
        print("This folder does not exists")
        return

    folder = Path(folder)
    data = np.load(os.path.join(folder, "results.npz"), allow_pickle=True)
    return data

def read_minimizing(dict) -> np.lib.npyio.NpzFile:
    """Function to return the minimizing_values.npz file for a specific job and task id."""

    save_dir = Path(file_path)
    folder = os.path.join(save_dir, "data/", dict)
    folder = Path(folder)
    if os.path.isdir(folder)==False:
        print("This folder does not exists")
        return

    aux = np.load(os.path.join(folder, "minimizing_values.npz"), allow_pickle=True)
    return aux

def calculate_gradients(beta_values, A, y, batches = None):
    """Function to calculate the gradient of the loss function w.r.t. beta for an array of beta_values.
    As this is recurrently needed in the evaluation process (we opted to not save this quantity for storage space reasons)

    beta_values     ---     list of np.arrays of size N
    A               ---     np.array size (M, N)
    y               ---     np.array size M
    batches         ---     optional for stochastic gradients w.r.t. that size,
                            if included should be a list of batches (of size k)
    """

    if batches is None:
        ATA = np.matmul(np.transpose(A), A)
        ATy = np.matmul(np.transpose(A), y)
        M = len(y)
        return [nf.gradient(beta, ATA, ATy, M) for beta in beta_values]
    else:
        return [nf.gradient_SGD(beta, indices, A, y) for beta, indices in zip(beta_values, batches)]


def eval(dir, alpha=None, show=False):
    """ Function creating the plots entries and loss_Bregman_H_cummulative for that specific save_dir
     Optionally one can filter for only specific alpha values"""

    folder = os.path.join(file_path, 'data/', dir)
    folder = Path(folder)

    if os.path.isdir(folder)==False:
        print("This folder does not exists")
        return

    print('I read the data')
    data = read_npz(dir)
    config = read_config_file(dir)


    print("The config file is:")
    print(config)

    if alpha is not None and config['alpha']!=alpha:
        return

    L = int(config['L'])
    eta = config['learning_rate']
    SGD = config['SGD']

    x = data['x']
    y = data['y']
    A = data['A']

    iterates = data['iterates'] #x iterates
    losses = data['losses']

    if eta=='adaptive':
        eta_values = data['step_sizes']


    if SGD:
        batches = data['batches']
        losses_batch = data['losses_batch']
        #gradient_norms_batch = [nf.gradient_SGD_L(iterate, batch, A, y, L) for iterate, batch in zip(iterates, batches)]

    #Calculate beta
    if config['difference']:
        beta_values = [it[0]**L - it[1]**L for it in iterates]
        sol_plus    = np.where(x>0, x, 0)
        sol_minus   = np.where(x<0, -x, 0)
        bregman_values = [hf.Bregman_distance(sol_plus, it[0]**L, L) + hf.Bregman_distance(sol_minus, it[1]**L, L) for it in iterates]
        if SGD:
            gradient_values = calculate_gradients(beta_values, A, y, batches)
        else:
            gradient_values = calculate_gradients(beta_values, A, y)
        if eta == 'adaptive':
            H_values = [[hf.H(eta_t, it[0] ** L, grad, L), hf.H(eta_t, it[1] ** L, -grad, L)] for eta_t, it, grad in
                        zip(eta_values, iterates, gradient_values)]
        else:
            H_values = [[hf.H(eta, it[0] ** L, grad, L), hf.H(eta, it[1] ** L, -grad, L)] for it, grad in
                        zip(iterates, gradient_values)]
        H_values = np.array(H_values)
        H_values_norm_u = np.linalg.norm(H_values[:, 0, :], axis=1)
        H_values_norm_v = np.linalg.norm(H_values[:, 1, :], axis=1)
        H_values_accumulated_u = np.cumsum(H_values_norm_u)
        H_values_accumulated_v = np.cumsum(H_values_norm_v)

    else:
        print('I calculate the iterates')
        beta_values = [it**L for it in iterates]
        # Calculate Bregman divergence
        bregman_values = [hf.Bregman_distance(x, beta, L) for beta in beta_values]
        if SGD:
            gradient_values = calculate_gradients(beta_values, A, y, batches)
        else:
            gradient_values = calculate_gradients(beta_values, A, y)
        if eta=='adaptive':
            H_values = [hf.H(eta_t, beta, grad, L) for eta_t, beta, grad in zip(eta_values, beta_values, gradient_values)]

        else:
            H_values = [hf.H(eta, beta, grad, L) for beta, grad in zip(beta_values, gradient_values)]
        H_values_norm = np.linalg.norm(np.array(H_values), axis=1)
        H_values_accumulated = np.cumsum(H_values_norm)

    print('I create or go in the saving directory.')
    save_dir = os.path.join(file_path, 'plots/', dir)
    os.makedirs(save_dir, exist_ok=True)
    save_dir = Path(save_dir)

    print('I start creating the plots.')

    plots.entries(beta_values, x, 1, folder_name=save_dir, file_prefix='', log=False, show=show)

    if config["difference"]:
        plots.Loss_vs_Bregman_vs_H(losses, bregman_values, H_values_accumulated_u, H_values_accumulated_v, log_x=False, log_y=True, folder_name=save_dir, file_prefix='', show=show)

    else:
        plots.Loss_vs_Bregman_vs_H(losses, bregman_values, H_values_accumulated, log_x=False, log_y=True, folder_name=save_dir, file_prefix='', show=show)

    plots.Loss_vs_Bregman_vs_H(losses, folder_name=save_dir, file_prefix='', show=show)
    plots.Loss_vs_Bregman_vs_H(losses, folder_name=save_dir, log_y=False, file_prefix='', show=show)

    if eta=='adaptive':
        plots.eta_values(eta_values, folder_name=save_dir, show=show)

    if SGD:
        plots.loss_compare_GD_SGD(losses, losses_batch, folder_name=save_dir)
        plots.loss_compare_GD_SGD(losses, losses_batch, y_scale=True, folder_name=save_dir)

    print('I am done with all the plots')

def calculate_minimzing_values(dict_name: str, task_id: int, alpha=None):
    """Calculates the quantities \beta_0_tilde, \delta and /sum_s H_s for the job_id, task_id pair
     saves it in the same directory.

     If alpha!= False it should be a float and only experiments with that initial value alpha will be considered.
    """

    print("Task id:", task_id)
    dict = (str(dict_name) + '/' + f"task_{task_id}")
    folder = os.path.join(file_path, "data/", dict)
    folder = Path(folder)

    if os.path.isdir(folder)==False:
        print("This folder does not exists")
        return

    data = read_npz(dict)
    config = read_config_file(dict)

    print("The config file is:")
    print(config)

    L = int(config['L'])
    SGD_bool = config['SGD']
    A = data['A']
    x = data['x']
    y = data['y']
    N = x.shape[0]


    iterates = data['iterates']

    eta = config['learning_rate']


    if alpha!=None and alpha!=config['alpha']:
        return


    print('I have all the data and calculate beta_values and gradients now')
    if config['difference']:
        beta_values = [it[0] ** L - it[1] ** L for it in iterates]
        u_0 = float(config['alpha'])**(1/L) * np.ones(N)
        v_0 = float(config['alpha'])**(1/L) * np.ones(N)
    else:
        beta_values = [it ** L for it in iterates]
        beta_0 = float(config['alpha']) * np.ones(N)

    if SGD_bool:
        batches = data['batches']
        gradients = calculate_gradients(beta_values, A, y, batches)
    else:
        gradients = calculate_gradients(beta_values, A, y)

    print('I start calculating H, delta and beta_0_tilde now')

    if config['difference']:
        if eta == 'adaptive':
            eta_values = data['step_sizes']
            H_values = [[hf.H(eta_t, it[0] ** L, grad, L), hf.H(eta_t, it[1] ** L, -grad, L)] for eta_t, it, grad in
                        zip(eta_values, iterates, gradients)]
        else:
            H_values = [[hf.H(eta, it[0] ** L, grad, L), hf.H(eta, it[1] ** L, -grad, L)] for it, grad in
                        zip(iterates, gradients)]

        H_values = np.array(H_values)
        delta_u = hf.delta(np.sum(H_values[:, 0, :], axis=0),  L)
        delta_v = hf.delta(np.sum(H_values[:, 1, :], axis=0),  L)

        u_0_tilde = hf.perturbed_init_uv(u_0, delta_u, L)
        v_0_tilde = hf.perturbed_init_uv(v_0, delta_v, L)

        print('I save everything now')
        minimizer_values(folder, [u_0_tilde, v_0_tilde], [delta_u, delta_v], [np.sum(H_values[:, 0, :], axis=0), np.sum(H_values[:, 1, :], axis=0)])
    else:
        if eta == 'adaptive':
            eta_values = data['step_sizes']
            H_values = [hf.H(eta_t, beta, grad, L) for eta_t, beta, grad in
                            zip(eta_values, beta_values, gradients)]
        else:
            H_values = [hf.H(eta, beta, grad, L) for beta, grad in zip(beta_values, gradients)]

        H_values = np.array(H_values)

        delta = hf.delta(np.sum(H_values, axis=0),  L)
        beta_0_tilde = hf.perturbed_init(beta_0, L, delta)

        print('I save everything now')
        minimizer_values(folder, beta_0_tilde, delta, np.sum(H_values, axis=0))

def eval_wrt_eta(dict_name, task_ids, alpha: float , show=False):
    """"""

    dict = str(dict_name)+'/'+f"task_{task_ids[0]}"
    config = read_config_file(dict)
    data = read_npz(dict)

    L = config['L']
    x = data['x']
    N = x.shape[0]

    if config['difference']:
        u_0 = alpha**(1/L)*np.ones(N)
        v_0 = alpha**(1/L)*np.ones(N)
    else:
        beta_0 = alpha * np.ones(N)

    diff_beta_0_l2 = []
    H_sum_l1 = []
    deltas = []

    eta_list = []


    for task_id in task_ids:
        task_id = int(task_id)
        print('Task id: ', task_id)

        #Go to the dictionary where the data is stored
        dict = (str(dict_name) + '/'+f"task_{task_id}")
        folder = os.path.join(file_path, "data/", dict)
        folder = Path(folder)
        if os.path.isdir(folder) == False:
            print("This folder does not exists")
            continue

        config = read_config_file(dict)
        print(config)

        if config['alpha']!=alpha:
            continue
        if config['learning_rate']=='adaptive':
            continue

        eta_list.append(float(config['learning_rate']))
        path = folder / "minimizing_values.npz"

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run '--eval_type minimizing' "
                f"for task {task_id} before evaluating it."
            )
        minimizing_values = read_minimizing(dict)

        if config['difference']:
            temp = minimizing_values['beta_0']
            diff_beta_0_l2.append([np.linalg.norm(u_0 - temp[0], ord=2), np.linalg.norm(v_0 - temp[1], ord=2)])
            deltas.append([np.linalg.norm(minimizing_values['delta'][0], ord=2), np.linalg.norm(minimizing_values['delta'][1], ord=2)])
            H_sum_l1.append([np.linalg.norm(minimizing_values['sum_H'][0], ord=2), np.linalg.norm(minimizing_values['sum_H'][1], ord=1)])
        else:
            temp = np.linalg.norm(beta_0 - minimizing_values['beta_0'], ord=2)
            diff_beta_0_l2.append(temp)
            deltas.append(np.linalg.norm(minimizing_values['delta'], ord=2))
            H_sum_l1.append(np.linalg.norm(minimizing_values['sum_H'], ord=1))

    print('Done with the data collection now, we start with the plots')

    folder_name = dict_name# f"{dict_name}/eta"

    save_dir = Path(file_path) / "plots" / folder_name
    os.makedirs(save_dir, exist_ok=True)
    save_dir = Path(save_dir)
    if config['difference']:
        diff_beta_0_l2 = np.array(diff_beta_0_l2)
        print(diff_beta_0_l2.shape)
        plots.with_respect_to_step_size(eta_list, diff_beta_0_l2, r"$\|\tilde{u}_0-u_0\|_2$", 'diff-beta-0-l2', folder_name=save_dir, show=show)
        #plots.with_respect_to_step_size(eta_list, diff_beta_0_l2[:,1], r"$\|\tilde{u}_0-u_0\|_2$", 'diff-beta-0-l2', folder_name=save_dir, show=True)
    else:
        plots.with_respect_to_step_size(eta_list, diff_beta_0_l2, r"$\|\tilde{\beta}_0-\beta_0\|_2$", 'diff-beta-0-l2', folder_name=save_dir, show=show)

    plots.with_respect_to_step_size(eta_list, H_sum_l1, r"$\|\sum_s H(\eta, \beta_s, \nabla L(\beta_s))\|_1$", name='H_sum_l1', folder_name=save_dir, show=show)
    plots.with_respect_to_step_size(eta_list, deltas, r"$\|\delta\|_2$", name='deltas', folder_name=save_dir, show=show)

def epsilon_evaluation(dict_name, task_ids, alpha=False, eta=False, show=False):
    dict = str(dict_name) + '/' + f"task_{task_ids[0]}"
    config = read_config_file(dict)
    data = read_npz(dict)

    L = config['L']
    x = data['x']
    N = x.shape[0]

    if config['difference']:
        return
    eps_list = []
    if alpha:
        eta_list = []
    elif eta:
        alpha_list = []


    for task_id in task_ids:
        task_id = int(task_id)
        print('Task id: ', task_id)
        dict = str(dict_name) + '/' + f"task_{task_id}"
        folder = os.path.join(file_path, "data/", dict)
        folder = Path(folder)
        if os.path.isdir(folder) == False:
            print("This folder does not exists")
            continue

        config = read_config_file(dict)

        if alpha and config['alpha']!=alpha:
            continue
        if config['learning_rate']=='adaptive':
            continue
        if eta and config['learning_rate']!=eta:
            continue
        if alpha:
            eta_list.append(float(config['learning_rate']))
        elif eta:
            alpha_list.append(float(config['alpha']))
        path = folder / "minimizing_values.npz"

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run '--eval_type minimizing' "
                f"for task {task_id} before evaluating it."
            )
        minimizing_values = read_minimizing(dict)
        if config['difference']:
            return
        else:
            beta_0_tilde = minimizing_values['beta_0']
            Q = np.linalg.norm(beta_0_tilde**((2-L)/L), ord=1)
            eps_list.append(float(hf.epsilon(beta_0_tilde, Q, L)))

    folder_name = f"{dict_name}#/Figure_7_8"
    save_dir = Path(file_path) / "plots" / folder_name
    os.makedirs(save_dir, exist_ok=True)
    save_dir = Path(save_dir)
    if eta:
        file_prefix = 'alpha_'
        plots.with_respect_to_step_size(alpha_list, eps_list, r"$\epsilon$", name=file_prefix + 'Epsilon', x_label=r'Initialization $\alpha$', x_scale='log', folder_name=save_dir, show=show)
    elif alpha:
        file_prefix = 'eta_'
        plots.with_respect_to_step_size(eta_list, eps_list, r"$\epsilon$", name=file_prefix + 'Epsilon', y_lim = 0.06, folder_name=save_dir, show=show)


def speed_of_convergence(dict_name, task_ids, alpha, show=True):
    loss_values = []
    eta_list = []


    for task_id in task_ids:
        task_id = int(task_id)
        dict = str(dict_name) + '/' + f"task_{task_id}"
        folder = os.path.join(file_path, "data/", dict)
        folder = Path(folder)
        if os.path.isdir(folder) == False:
            print("This folder does not exists")
            continue
        config = read_config_file(dict)

        if config['alpha'] != alpha:
            continue
        if config['learning_rate'] in [0.15, 0.25, 0.35, 0.45]:
            continue

        data = read_npz(dict)
        if config['learning_rate'] == 'adaptive':
            continue

        losses = data['losses']
        loss_values.append(losses)


        eta_list.append(config['learning_rate'])


    folder_name = dict_name #+ '/Figure_3'

    save_dir = Path(file_path) / "plots" / folder_name
    os.makedirs(save_dir, exist_ok=True)
    save_dir = Path(save_dir)

    plots.losses_convergence_speed(eta_list, loss_values, 'convergence_speed', folder_name=save_dir, show=show)

def adaptive_step_size(dict_name, task_id=None, show=True):
    if task_id!=None:
        task_id = int(task_id)
        dict = str(dict_name) + '/' + f"task_{task_id}"
    else:
        dict = str(dict_name)
    folder = os.path.join(file_path, "data/", dict)
    folder = Path(folder)
    if os.path.isdir(folder) == False:
        print("This folder does not exists")
        return
    config = read_config_file(dict)

    if config['learning_rate'] != 'adaptive':
        print("This call is only for evaluating adaptive step sizes.")
        return

    data = read_npz(dict)

    eta_values = data['step_sizes']
    folder_name = dict_name #+ '/Figure_4'
    save_dir = Path(file_path) / "plots" / folder_name
    os.makedirs(save_dir, exist_ok=True)
    save_dir = Path(save_dir)
    plots.eta_values(eta_values, folder_name=save_dir, show=show)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--eval_type", choices=["example", "minimizing", "eta", "speed", "epsilon", "adaptive"], required=True)
    p.add_argument("--tasks", type=int, nargs="+", required=False)
    p.add_argument("--alpha", type=float, required=False)
    p.add_argument("--eta", type=float, required=False)
    args = p.parse_args()
    show = True
    if not args.output_dir.strip():
        p.error("--output_dir must not be empty.")

    folder = os.path.join(file_path, 'data/', args.output_dir)
    if os.path.isdir(folder)==False:
        p.error(f"Results directory does not exist: {args.output_dir}")

    if args.alpha is not None:
        if args.alpha <= 0:
            p.error("--alpha must be greater than 0.")

    if args.eta is not None:
        if args.eta <= 0:
            p.error("--eta must be greater than 0.")

    # Every mode except 'example' operates on task subdirectories.
    if args.eval_type != "example" and args.eval_type != "adaptive":
        if args.tasks is None:
            p.error(
                f"--tasks is required for --eval_type {args.eval_type}."
            )

        if any(task < 0 for task in args.tasks):
            p.error("--tasks must contain nonnegative integers.")

        if len(args.tasks) != len(set(args.tasks)):
            p.error("--tasks must not contain duplicate indices.")

    if args.eval_type == "epsilon":
        # True when both are absent or both are present.
        if (args.alpha is None) == (args.eta is None):
            p.error(
                "'epsilon' requires exactly one of --alpha or --eta."
            )

    if args.eval_type == 'example':
        eval(args.output_dir, show=show)
    elif args.eval_type == 'minimizing':
        if tasks is None:
            calculate_minimzing_values(args.output_dir, alpha=args.alpha)
        else:
            for i in args.tasks:
                calculate_minimzing_values(args.output_dir, i, alpha=args.alpha)
    elif args.eval_type == 'eta':
        eval_wrt_eta(args.output_dir, args.tasks, alpha=args.alpha, show=show)
    elif args.eval_type == 'speed':
        speed_of_convergence(args.output_dir, args.tasks, args.alpha, show=show)
    elif args.eval_type == 'epsilon':
        epsilon_evaluation(args.output_dir, args.tasks, eta=args.eta, alpha=args.alpha, show=show)
    elif args.eval_type == 'adaptive':
        if args.tasks is None:
            adaptive_step_size(args.output_dir, show=show)
        else:
            for task in args.tasks:
                adaptive_step_size(args.output_dir, task, show=show)
    else:
        p.error("--eval must be one of [example, minimizing, eta, speed, epsilon, 'adaptive'].")


if __name__ == '__main__':
    main()