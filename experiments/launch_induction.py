from experiments.launcher import KubernetesJob, launch
import subprocess
import argparse
import numpy as np
import os

CPU = 4

def main(
    testing: bool,
    is_adria: bool,
):
    # thresholds = 10 ** np.linspace(-2, 0.5, 21)
    thresholds = [0.5623]
    seed = 424671755

    commands: list[list[str]] = []
    for reset_network in [0, 1]:
        for zero_ablation in [0, 1]:
            for loss_type in ["kl_div"]:
                for threshold in [1.0] if testing else thresholds:
                    command = [
                        "python",
                        "acdc/main.py" if (not is_adria) else "/Automatic-Circuit-Discovery/acdc/main.py",
                        "--task=induction",
                        f"--threshold={threshold:.5f}",
                        "--using-wandb",
                        f"--wandb-run-name=launch_induction-acdc-{len(commands):03d}",
                        "--wandb-group-name=acdc-induction",
                        f"--device=cuda",
                        f"--reset-network={reset_network}",
                        f"--seed={seed}",
                        f"--metric={loss_type}",
                        f"--torch-num-threads={CPU}",
                        "--wandb-dir=wandb_runs/",
                        f"--wandb-mode={'offline' if testing else 'online'}",
                    ]
                    if zero_ablation:
                        command.append("--zero-ablation")

                    commands.append(command)
    for command in commands:
        print("-----------")
        print("Running", " ".join(command))
        
    if is_adria:
        launch(
            commands,
            name="acdc-induction",
            job=None
            if testing
            else KubernetesJob(container="ghcr.io/rhaps0dy/automatic-circuit-discovery:1.2.8", cpu=CPU, gpu=0),
        )

    else:
        for command in commands:
            print("Running", command)
            subprocess.run(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    print("os.getcwd():", os.getcwd())
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--is-adria", action="store_true")
    main(
        testing=parser.parse_args().testing,
        is_adria=parser.parse_args().is_adria,
    )
