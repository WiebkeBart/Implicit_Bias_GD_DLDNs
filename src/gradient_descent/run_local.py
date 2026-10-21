import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    # Forward remaining arguments to the experiment program.
    args, experiment_args = parser.parse_known_args()

    for task_id in args.tasks:
        output_dir = args.output_dir / f"task_{task_id}"

        command = [
            sys.executable,
            "-m", "gradient_descent.main",
            str(task_id),
            "--output_dir", str(output_dir),
            *experiment_args,
        ]

        print(f"Starting task {task_id}: {output_dir}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()