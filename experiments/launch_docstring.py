from experiments.launcher import KubernetesJob, launch
import numpy as np

CPU = 4


def main(testing: bool):
    thresholds = 10 ** np.linspace(-2, 0.5, 21)
    # thresholds = [0.095]
    seed = 516626229

    commands: list[list[str]] = []
    for reset_network in [0,1]:
        for zero_ablation in [0, 1]:
            for loss_type in ["kl_div","docstring_metric"]: # "docstring_stefan", "nll", "match_nll" , ""docstring_metric""
                for threshold in [1.0] if testing else thresholds:
                    command = [
                        "python",
                        "acdc/main.py",
                        "--task=docstring",
                        f"--threshold={threshold:.5f}",
                        "--using-wandb",
                        f"--wandb-run-name=launch_docstring-acdc-{len(commands):03d}",
                        "--wandb-group-name=acdc-docstring",
                        f"--device=cuda",
                        f"--reset-network={reset_network}",
                        f"--seed={seed}",
                        f"--metric={loss_type}",
                        f"--torch-num-threads=0",
                        "--wandb-dir=wandb_runs/",  # If it doesn't exist wandb will use /tmp
                        f"--wandb-mode={'offline' if testing else 'online'}",
                        f"--max-num-epochs={1 if testing else 100_000}",
                        "--first-cache-cpu=False",
                        "--second-cache-cpu=False"
                    ]
                    if zero_ablation:
                        command.append("--zero-ablation")

                    commands.append(command)
    for command in commands:
        # print("-----------")
        print(" ".join(command))
    # launch(
    #     commands,
    #     name="acdc-docstring",
    #     job=None
    #     if testing
    #     else KubernetesJob(container="ghcr.io/rhaps0dy/automatic-circuit-discovery:1.2.10", cpu=CPU, gpu=0),
    # )


if __name__ == "__main__":
    main(testing=False)
