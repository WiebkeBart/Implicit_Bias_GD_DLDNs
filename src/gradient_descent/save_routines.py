import csv
from tempfile import NamedTemporaryFile
from pathlib import Path
import os
import numpy as np
import json


def save_config(folder, seed, learning_rate, alpha, difference, L, eps, T, SGD_bool = False, batch_size = None):
    """Function to write the config file in that specific folder"""
    config = {
        "seed": seed,
        "L": L,
        "difference": difference,
        "learning_rate": learning_rate,
        "alpha": alpha,
        "SGD": SGD_bool,
        "batch_size": batch_size,
        "stopping": {
            "criterion": "full_loss",
            "tolerance": eps,
            "max_iterations": T
        },
    }

    print(f'The config file: {config}\n is saved in folder: {folder}')

    with open(folder / "config.json", "w") as file:
        json.dump(config, file, indent=2)


def save_data(folder, x_values, loss_values, gradient_norm, step_sizes, A, x, y, loss_batch=None, batches=None):
    """Function to save the data in a npz file

    I opted not to save the full gradient, but rather to reconstruct it later from the data,
    for this save A, x, y then for SGD we need the batches."""
    print(f'The data is saved in folder: {folder}')

    np.savez_compressed(
        folder/ "results.npz",
        iterates=np.asarray(x_values),
        losses=np.asarray(loss_values),
        losses_batch=np.asarray(loss_batch),
        gradient_norms=np.asarray(gradient_norm),
        step_sizes=np.asarray(step_sizes),
        A=np.asarray(A),
        x=x,
        y=y,
        batches=np.asarray(batches)
    )

def minimizer_values(folder, beta_0_tilde, delta, sum_H):
    """Function to save the values of the optimization problem in folder.
    The objects are:
        beta_0_tilde        -   np.array // in the general case [u_0_tilde, v_0_tilde]
        delta               -   np.array // in the general case [delta_u, delta_v]
        sum_H               -   np.array // in the general case [sum_H_u, sum_H_v]"""
    np.savez_compressed(
        folder/ "minimizing_values.npz",
        beta_0=beta_0_tilde,
        delta=delta,
        sum_H=sum_H
    )

