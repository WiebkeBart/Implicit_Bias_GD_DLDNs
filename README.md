# (Stochastic) Gradient Descent for Deep Linear Diagonal Networks

Code accompanying the paper:

**Implicit Bias of (Stochastic) Gradient Descent on Deep Linear Diagonal Networks**

by Wiebke Bartolomaeus and Holger Rauhut 
[paper link, and publication details will be made available once at hand]

This project studies gradient descent (GD) and stochastic gradient
descent (SGD) on synthetic sparse-recovery problems using deep diagonal
networks.
It allows to reproduce the plots presented in the aforementioned paper.

## Installation

Requirements:
- Python ≥3.11.8. The reference experiments were tested with Python 3.11.8.
- uv, if not already installed check [here](https://docs.astral.sh/uv/getting-started/installation/)
- Slurm, this is only needed for cluster execution


Clone the repository and select the release associated with the paper:

```bash
git clone https://github.com/WiebkeBart/Implicit_Bias_GD_DLDNs.git
cd Gradient_Descent
git checkout first-version
uv sync --locked
```
`uv sync --locked` creates the local environment using the recorded dependencies.
The dependencies are declared in `pyproject.toml`. Their resolved
versions are recorded in `uv.lock`.

Run all commands below from the repository root.

## Run a single experiment

For example, run the first parameter combination locally:

```bash
uv run --locked python -m gradient_descent.main 62 \
    --seed 23 \
    --N 100 \
    --M 20 \
    --s 3 \
    --L 3 \
    --output_dir local/example_001
```

This command saves results in `data/local/example_001/. The program adds the `data` prefix automatically.

Arguments:

| Argument | Meaning |
|---|---|
| `param_idx` | Index of the parameter combination in the experiment grid. Positional argument, not optional  |
| `--seed` | Random seed |
| `--N` | Dimension of the target vector |
| `--M` | Number of measurements |
| `--s` | Sparsity of the target vector |
| `--L` | Network depth |
| `--SGD` | Use stochastic gradient descent |
| `--batch_size` | Batch size for SGD, does not have any effect for GD |
| `--diff` | Use the difference parameterization |
| `--output_dir` | Directory for this experiment's results. Should be unique for each experiment |

Without `--SGD`, the experiment uses gradient descent.
To run SGD, add:

```bash
--SGD --batch_size k
```

Choose a batch size $k$ between $1$ and $M$. Use a separate output directory
for each experiment: repeating a run in the same directory overwrites
its saved configuration and results.

Display help:

```bash
uv run --locked python -m gradient_descent.main --help
```

By default, experiments use what we call positive model in the paper, i.e. $\beta = u^{\odot L}$.  
With `--diff`, they use the difference parameterization, i.e. $\beta = u^{\odot L} - v^{\odot L}$, discussed in the general-case section of the paper. 
Powers are taken elementwise.

The experiment index selects a learning rate and initialization from
the parameter grid defined in `src/gradient_descent/main.py`.

## Parameter grid

The grid is defined in `src/gradient_descent/main.py`:

```python
eta_values = [
    0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2,
    0.1, 0.05, 0.01, 0.005, 0.001, "adaptive"]

alpha_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
```

There are 78 combinations. Valid indices for param_idx are hence 0–77.


The index is interpreted as:

```python
alpha = alpha_values[param_idx // 13]
eta = eta_values[param_idx % 13]
```

The value `adaptive` for eta corresponds to the adaptive step-size strategies, discussed in Remark 2, Remark 4 and Remark 5 for the cases of gradient descent on the positive mode, general model and sgd respectively in the paper.

| Indices | Initialization |
|---|---|
| 0–12 | `1e-9` |
| 13–25 | `1e-8` |
| 26–38 | `1e-7` |
| 39–51 | `1e-6` |
| 52–64 | `1e-5` |
| 65–77 | `1e-4` |

Within each group, learning rates follow the order listed above.
Changing the grid changes the meaning of the indices.

## Outputs

Each experiment writes:

- `config.json`: Seed, model and optimization settings, and stopping criterion
- `results.npz`: Generated data, iterates, losses, gradient norms, step sizes, and SGD batches

## Run on a Slurm cluster

Running on a cluster is very useful for running multiple experiments in parallel. 

Prepare the environment on the cluster:

```bash
uv sync --locked
```

Review the resource settings and experiment arguments in `run.sh` -- by considering the specifications below under Reproduce the paper's results --
then submit:

```bash
sbatch run.sh
```

`#SBATCH --array` specifies with which param_idx the main module will be called. 

For example 
```bash
    #SBATCH --array=52-64%10
```
runs 13 experiments with `alpha=1e-5` and all possible eta values. 
%10 allows at most ten array tasks to run concurrently; the scheduler may run fewer depending on available resources.

The experiments will be saved in individual subfolders of the folder named by the SLURM_ARRAY_JOB_ID. The subfolders are labeled according to the entries in the array. 

In our example this would be
```text
data/job_{SLURM_ARRAY_JOB_ID}/
├── task_52/
├── task_53/
├── task_54/
├── task_55/
└── task_56/
....
└── task_64/
```

## Multiple runs locally

The module `gradient_descent.run_local` runs several parameter
combinations sequentially on your local machine.

Use `--tasks` to supply a list of parameter indices instead of the
single positional `param_idx` argument. For the current parameter grid,
valid indices are 0–77. All other experiment arguments are forwarded
to each run and remain the same across the batch. 
Therefore, each task uses the same seed and other shared arguments; the task index changes the selected initialization and learning rate.

For example:

```bash
uv run --locked python -m gradient_descent.run_local \
    --tasks 59 60 61 62 63 \
    --output_dir local/sweep \
    --seed 23 --L 3 --N 100 --M 20 --s 3
```

This runs five experiments with initialization `alpha=1e-5` and
learning rates `0.1`, `0.05`, `0.01`, `0.005`, and `0.001`.

Results are saved in separate task directories:

```text
data/local/sweep/
├── task_59/
├── task_60/
├── task_61/
├── task_62/
└── task_63/
```

Every task directory contains its own `config.json` and `results.npz`.
The program adds the `data/` prefix automatically, so supply
`--output_dir local/sweep`, not `--output_dir data/local/sweep`.

Experiments run one after another, in the order supplied. The runner
stops if an experiment exits with an error. It does not automatically
skip previously completed experiments; rerunning a task in the same
directory overwrites its saved results.

For large sweeps, consider cluster execution with a Slurm job array,
which can run multiple experiments concurrently when resources are
available.


## Evaluation 
 
Evaluation reads saved experiment results and creates plots or derived quantities. The --eval_type argument selects the analysis.

To evaluate the single experiment shown above:

```bash
uv run --locked python -m gradient_descent.evaluation \
    --output_dir local/example_001 --eval_type 'example'
```


Arguments:

| Argument | Meaning |
|---|---|
|  `--output_dir` | Directory where the experiment's results lie. |
| `--eval_type` | specifies which kind of evaluation should be performed. Explained below. |
| `--tasks` | Multiple numbers if multiple experiments should be considered. They must lie in subfolders indexed by the task number.| 
| `--alpha` | Possiblity to filter for a specific alpha value. |
| `--eta` | Possibility to filter for a specific eta value. |


Possible values for `--eval_type`:


| Argument | Purpose | Required input |
|---|---|---|
|`example` | creates plots for a single experiment, consisting of the iterates, the Loss and Bregman Divergence w.r.t. iterates. | corresponding experiment data needs to be available.|
|`minimizing`| creates the `minimizing_values.npz` in the `--output_dir` folder, if called with `--tasks` in each subfolder. | corresponding experiment data needs to be available. |
|`eta` | create plots w.r.t. different step-size, for a fixed initialization. |needs several `--tasks`  and `--alpha` to filter, needs `minimizing` to be run beforehand|
|`speed`| plots convergence speed for different step-sizes, for a fixed initialization. |needs several `--tasks` and `--alpha` to filter|
|`adaptive`| plots the step-sizes choosen by the adaptive strategy for one run| can be used with single integer`--tasks` or without.|
|`epsilon`| Shows the evaluation of $\varepsilon$ defined in the paper (65) either w.r.t. step size for a fixed initalization or w.r.t. the initialization for a fixed step size. |needs several `--tasks` and **one** of `--alpha` or `--eta` to filter, needs `minimizing` to be run beforehand |



The specification `minimizing` creates the following output in `data/output_dir/task_idx` for each idx in `--tasks`
If `--tasks` is not selected it will write in `data/output_dir`
- `minimizing_values.npz`: values related to the perturbed initialization.


## Reproduce the paper's results

Fixed for all experiments are the following parameters:

    `--seed 23 --L 3 --N 100 --M 20 --s 3`

If not specified otherwise. The run for experiment configuration is `gradient_descent.main` and for the evaluation `gradient_descent.evaluation`.
Below we only specify the parameters specific for that task and if another method then the just mentioned should be called.
The values for `--output_dir` are just recommendations. 
For Slurm runs, evaluate the parent directory job_<array-job-id> with the corresponding --tasks. For a single-run evaluation, use job_<array-job-id>/task_<index>.


Evaluation config should have the same `--output_dir` and `--tasks` as the corresponding values in Experiment configuration

| Figure or table | Experiment configuration | Evaluation config |
|---|---|---|
| Figure 1 + 2 | param_idx:62, --output_dir 'local/figure1+2' | --eval_type example |
| Figure 3  | gradient_descent.run_local --tasks 52 53 54 55 56 57 58 59 60 61 62 63  --output_dir 'local/GD_pos' | --eval_type speed --alpha 1e-5  |
| Figure 4 | gradient_descent.run_local --tasks 64 --output_dir 'local/GD_pos' | --eval_type adaptive|
| Figure 5 + 6 | same as Figure 3 | --eval_type minimizing and --eval_type eta with --alpha 1e-5 |
| Figure 7 | same as Figure 3| --eval_type minimizing and --eval_type epsilon --alpha 1e-5   |
| Figure 8 | gradient_descent.run_local --tasks 10 23 36 49 62 75  --output_dir 'local/GD_pos'| --eval_type minimizing and --eval_type epsilon --eta 0.005|
| Figure 9 + 10 | param_idx:62 --output_dir 'local/figure9+10' --diff | --eval_type example |
| Figure 11 + 12 | same as in Figure 3, except --output_dir 'local/GD_gen' --diff | --eval_type minimizing and --eval_type eta with --alpha 1e-5|
| Figure 13 | param_idx:62 --output_dir 'local/figure13' --SGD --batch_size 5| --eval_type example |
| Figure 14 |same as Figure 3, except --output_dir 'local/SGD_pos' --SGD --batch_size 5| --eval_type minimizing and --eval_type eta with --alpha 1e-5|

Below you find the necessary commands to create each figure. 
Note that experiments will be rerun and overwritten if all of these are carried out. 
So it might be useful to only run the commands not run before. 

* Figure 1 & 2
```bash
    uv run --locked python -m gradient_descent.main 62 \
        --seed 23 --L 3 --N 100 --M 20 --s 3 \
        --output_dir 'local/GD_pos'
    uv run --locked python -m gradient_descent.evaluation \
        --output_dir 'local/GD_pos' \
        --eval_type example
```

* Figure 3 
```bash 
    uv run --locked python -m gradient_descent.run_local\
        --tasks 52 53 54 55 56 57 58 59 60 61 62 63 \
        --output_dir 'local/GD_pos' \
        --seed 23 --L 3 --N 100 --M 20 --s 3
    uv run --locked python -m gradient_descent.evaluation \
        --tasks 52 53 54 55 56 57 58 59 60 61 62 63 \
        --output_dir 'local/GD_pos' \
        --eval_type speed --alpha 1e-5    
 ```

* Figure 4
```bash
    uv run --locked python -m gradient_descent.run_local\
          --tasks 64 \
          --output_dir 'local/GD_pos' \
          --seed 23 --L 3 --N 100 --M 20 --s 3
    uv run --locked python -m gradient_descent.evaluation \
        --tasks 64 \
        --output_dir 'local/GD_pos' \
        --eval_type adaptive
```

* Figure 5 & 6
```bash
    uv run --locked python -m gradient_descent.run_local\
        --tasks 52 53 54 55 56 57 58 59 60 61 62 63 \
        --output_dir 'local/GD_pos' \
        --seed 23 --L 3 --N 100 --M 20 --s 3          
    uv run --locked python -m gradient_descent.evaluation \
        --output_dir  'local/GD_pos' \
        --tasks 52 53 54 55 56 57 58 59 60 61 62 63 \
        --eval_type minimizing
    uv run --locked python -m gradient_descent.evaluation \
        --output_dir  'local/GD_pos' \
        --tasks 52 53 54 55 56 57 58 59 60 61 62 63\
        --eval_type eta --alpha 1e-5

 ```
* Figure 7
```bash
    uv run --locked python -m gradient_descent.run_local \
        --tasks 52 53 54 55 56 57 58 59 60 61 62 63\
        --output_dir 'local/GD_pos' \
        --seed 23 --L 3 --N 100 --M 20 --s 3
    uv run --locked python -m gradient_descent.evaluation \
        --output_dir 'local/GD_pos' \
        --tasks 52 53 54 55 56 57 58 59 60 61 62 63\
        --eval_type minimizing
    uv run --locked python -m gradient_descent.evaluation \
        --output_dir  'local/GD_pos' \
        --tasks 52 53 54 55 56 57 58 59 60 61 62 63 \
        --eval_type epsilon --alpha 1e-5    
 ```
* Figure 8
```bash
    uv run --locked python -m gradient_descent.run_local \
        --tasks 10 23 36 49 62 75 \
        --output_dir 'local/GD_pos' \
        --seed 23 --L 3 --N 100 --M 20 --s 3     
    uv run --locked python -m gradient_descent.evaluation \
        --output_dir 'local/GD_pos' \
        --tasks 10 23 36 49 62 75\
        --eval_type minimizing
    uv run --locked python -m gradient_descent.evaluation \
        --tasks 10 23 36 49 62 75 \
        --output_dir 'local/GD_pos' \
        --eval_type epsilon --eta 0.005    
 ```
* Figure 9 & 10
```bash
  uv run --locked python -m gradient_descent.main 62 \
      --output_dir 'local/GD_gen' \
      --seed 23 --L 3 --N 100 --M 20 --s 3 --diff 
  uv run --locked python -m gradient_descent.evaluation \
      --output_dir 'local/GD_gen' --eval_type example
 ```
* Figure 11 & 12
```bash 
    uv run --locked python -m gradient_descent.run_local\
        --tasks 52 53 54 55 56 57 58 59 60 61 62 63 \
        --output_dir 'local/GD_gen' \
        --seed 23 --L 3 --N 100 --M 20 --s 3 --diff
    uv run --locked python -m gradient_descent.evaluation \
        --output_dir  'local/GD_gen' \
        --tasks 52 53 54 55 56 57 58 59 60 61 62 63 \
        --eval_type minimizing
    uv run --locked python -m gradient_descent.evaluation \
        --tasks 52 53 54 55 56 57 58 59 60 61 62 63 \
        --output_dir 'local/GD_gen' \
        --eval_type eta --alpha 1e-5    
 ```
* Figure 13
```bash
    uv run --locked python -m gradient_descent.main 62 \
        --output_dir 'local/SGD_pos' \
        --seed 23 --L 3 --N 100 --M 20 --s 3 --SGD --batch_size 5
    uv run --locked python -m gradient_descent.evaluation \
        --output_dir 'local/SGD_pos' \
        --eval_type example
```
* Figure 14
```bash
   uv run --locked python -m gradient_descent.run_local\
        --tasks 52 53 54 55 56 57 58 59 60 61 62 63 \
        --output_dir 'local/SGD_pos' \
        --seed 23 --L 3 --N 100 --M 20 --s 3 --SGD --batch_size 5
    uv run --locked python -m gradient_descent.evaluation \
        --output_dir 'local/SGD_pos' \
        --tasks 52 53 54 55 56 57 58 59 60 61 62 63 \
        --eval_type minimizing 
    uv run --locked python -m gradient_descent.evaluation \
        --output_dir 'local/SGD_pos' \
        --tasks 52 53 54 55 56 57 58 59 60 61 62 63\
        --eval_type eta --alpha 1e-5
```


## Reproducibility

The reference experiments were run with:

- Code release: 3b482f1a739010e6854048ff0056d435b26a3ee5
- Python: 3.11.8
- NumPy: 2.4.6
- Matplotlib: 3.11.2
- Operating system: macOS 15.6.1
- Hardware: Apple M3, 8 CPU cores, 24 GB RAM

Random seeds are recorded with the experiment settings.
`uv.lock` records the Python dependency versions used for this release.

Numerical results may differ slightly across operating systems,
hardware, and numerical-library implementations.


## Repository structure

```text
src/gradient_descent/
    DataGeneration.py   Generate synthetic data
    network_fcts.py     Network, loss, and gradient functions
    gradient_method.py  GD and SGD implementations
    main.py             Experiment entry point
    evaluation.py       Analyze saved experiments
    plots.py            Plotting functions
    help_fcts.py        Mathematical helper functions
    save_routines.py    Save experiment settings and results
    run_local.py

run.sh                  Slurm submission script
pyproject.toml          Project metadata and dependencies
uv.lock                 Resolved dependency versions
```

## Citation

If you use this code, please cite:

[Paper citation or BibTeX entry]
