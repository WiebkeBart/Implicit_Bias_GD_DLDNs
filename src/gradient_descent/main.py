import numpy as np
from . import DataGeneration
from . import network_fcts as nf
from . import plots
from .gradient_method import GD, SGD
import random
import os # for file saving
from pathlib import Path #for file saving
from .save_routines import save_config, save_data
import argparse
from . import save_routines


from .help_fcts import Bregman_distance, H
from . import help_fcts as hf



file_path = os.getcwd() #set the directory we are in

def single_run(seed, alpha, eta=0.01, N=100, M=20, s=3, L=3, difference=False, sgd_bool=False, batch_size=None, save=False, output_dir=None, output=True):
    eps = 1e-5
    T = int(10e7)

    # Save the experiment setup
    save_dir = os.path.join(file_path, 'data/', output_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_dir = Path(save_dir)
    save_config(save_dir, seed, eta, alpha, difference, L, eps, T, sgd_bool, batch_size)

    # setting the seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    A, x, y = DataGeneration.vector(N, M, s, positive=not difference, normalized=True)
    # Calculating useful quantities to minimize calculations
    ATA = np.matmul(A.T, A)
    yTA = np.matmul(y.T, A)
    ATy = yTA.T
    yTy = np.matmul(y.T, y)


    gradient_beta = lambda x: nf.gradient(x, ATA=ATA, ATy=ATy, M=M)
    loss_beta = lambda x: nf.Loss(x, ATA=ATA, yTA=yTA, yTy=yTy, M=M)

    # Initialization and definition of loss/gradient function in both cases
    if difference:
        u0 = np.power(alpha, 1 / L) * np.ones(N)
        v0 = u0
        x0 = np.array((u0, v0))

        if sgd_bool:
            gradient_batch_fct = lambda x, i: nf.gradient_uv_SGD(x, i, A, y, L)
            loss_batch = lambda x, i: nf.batch_loss_uv(x, i, A, y, L)

        gradient_fct = lambda x: nf.gradient_u_v(x,  ATA=ATA, ATy=ATy, L=L, M=M)
        loss_fct = lambda x: nf.Loss_uv(x, ATA=ATA, yTA=yTA, yTy=yTy, L=L, M=M)

    else:
        x0 = np.power(alpha, 1 / L) * np.ones(N)

        if sgd_bool:
            gradient_batch_fct = lambda x, i: nf.gradient_SGD_L(x, i, A, y, L)
            loss_batch = lambda x, i: nf.batch_loss_L(x, i, A, y, L)
        gradient_fct = lambda x: nf.gradient_L(x, ATA=ATA, ATy=ATy, L=L, M=M)
        loss_fct = lambda x: nf.Loss_L(x, ATA=ATA, yTA=yTA, yTy=yTy, L=L, M=M)

    if eta=='adaptive':
        if difference:
            sol_plus_2 = np.linalg.norm(np.maximum(x,0), ord=2)
            sol_minus_2 = np.linalg.norm(np.maximum(-x,0), ord=2)
            eta_fct = lambda x: hf.adaptive_step_size_difference(x[0], x[1], L, sol_plus_2, sol_minus_2, gradient_beta, loss_beta)
        else:
            sol_2 = np.linalg.norm(x, ord=2)
            eta_fct = lambda x: hf.adaptive_step_size(x, L, sol_2, gradient_beta, loss_beta)
    else:
        eta_fct = lambda x: eta

    #run of the Gradient Algorithm
    if sgd_bool:
        x_values, loss_values, loss_sgd_values, grad_norm, grad_values, eta_values, batches = SGD(x0, loss_fct, loss_batch, gradient_batch_fct, batch_size, M, eta_fct, eps=eps, output=False, T=T)
    else:
        x_values, loss_values, grad_norm, grad_values, eta_values = GD(x0, loss_fct, gradient_fct, eta_fct, eps=eps, output=output, T=T)
        batches = None
        loss_sgd_values = None


    save_data(save_dir, x_values, loss_values, grad_norm, eta_values, A, x, y, loss_sgd_values, batches)



### Here we specify the experiment we are going to run when executing
def main():
    p = argparse.ArgumentParser()
    p.add_argument("param_idx", type=int)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--M", type=int, required=True)
    p.add_argument("--s", type=int, required=True)
    p.add_argument("--SGD", action="store_true")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--diff", action="store_true")
    p.add_argument("--output_dir", type=str, required=True)
    args = p.parse_args()

    # Check plausibility of the arguments
    # Validate dimensions.
    if args.N <= 0:
        p.error("--N must be a positive integer.")

    if args.M <= 0:
        p.error("--M must be a positive integer.")

    # Validate sparsity.
    if not 1 <= args.s <= args.N:
        p.error(f"--s must be between 1 and N={args.N}.")
    # This assumes the supported model depths are L >= 2.
    if args.L < 2:
        p.error("--L must be at least 2.")

    # np.random.seed() accepts integers in this range.
    if not 0 <= args.seed <= 2 ** 32 - 1:
        p.error(f"--seed must be between 0 and {2 ** 32 - 1}.")

    if args.SGD and args.batch_size is None:
        p.error("--SGD requires --batch_size.")

    if args.batch_size is not None:
        if not 1 <= args.batch_size <= args.M:
            p.error(f"--batch_size must be between 1 and M={args.M}.")
    # Reject an empty or whitespace-only output directory.
    if not args.output_dir.strip():
        p.error("--output_dir must not be empty.")

    eta_values = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 'adaptive']
    alpha_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    n_eta = len(eta_values)
    n_alpha = len(alpha_values)

    if not 0 <= args.param_idx < n_eta*n_alpha:
        p.error(f"param_idx must be between 0 and {n_eta*n_alpha-1}.")

    idx_alpha = args.param_idx // n_eta
    idx_eta = args.param_idx % n_eta

    eta = eta_values[idx_eta]
    alpha = alpha_values[idx_alpha]

    print(args.seed, eta, alpha)
    print(f"Difference={args.diff}, SGD={args.SGD}")
    if args.SGD:
        print(f"batch_size={args.batch_size}")
    single_run(args.seed, alpha, eta, N=args.N, M=args.M, s=args.s, L=args.L, difference=args.diff, sgd_bool=args.SGD, batch_size=args.batch_size, save=True, output_dir=args.output_dir, output=False)


## Here one can run one experiment
if __name__ == '__main__':
    print('Main function')
    main()
