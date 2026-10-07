"""Modal entry point for Task 3 experiments; method runners live in experiments/.

Prepare the shared dataset splits first, then run:
  modal run modal_app.py --dataset all --action prepare
  modal run modal_app.py --dataset all --action sslae
  modal run modal_app.py --dataset all --action vime
  modal run modal_app.py --dataset all --action stunt
  modal run modal_app.py --dataset all --action gftab
  modal run modal_app.py --dataset all --action rankup
  modal run modal_app.py --dataset all --action all --max-parallel-jobs 4
The named Modal Volumes task3-data and task3-results persist inputs and outputs.
"""

import json
from pathlib import Path

import modal

app = modal.App("task3-ssl-baselines")
data_volume = modal.Volume.from_name("task3-data", create_if_missing=True)
results_volume = modal.Volume.from_name("task3-results", create_if_missing=True)


def add_experiment(image: modal.Image, name: str) -> modal.Image:
    return (
        image.add_local_file("experiments/__init__.py", remote_path="/opt/task3/experiments/__init__.py")
        .add_local_dir(f"experiments/{name}", remote_path=f"/opt/task3/experiments/{name}")
    )


prepare_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy<2", "pandas", "scikit-learn==1.3.2", "joblib")
    .add_local_dir("dataset", remote_path="/opt/task3/dataset")
    .add_local_file("prepare_data.py", remote_path="/opt/task3/prepare_data.py")
    .add_local_file("config/datasets.json", remote_path="/opt/task3/config/datasets.json")
)

sslae_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy<2", "pandas", "scikit-learn==1.3.2", "joblib",
        "torch==2.1.2", "pytorch-lightning==2.0.6", "rich", "torchmetrics==1.8.2",
    )
)
sslae_image = add_experiment(sslae_image, "sslae")

vime_image = (
    modal.Image.debian_slim(python_version="3.10")
    # The official Linux extra installs TensorFlow's matching CUDA/cuDNN wheels.
    .pip_install("numpy<2", "scikit-learn==1.3.2", "tensorflow[and-cuda]==2.15.1")
)
vime_image = add_experiment(vime_image, "vime")

stunt_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy<2", "scikit-learn==1.3.2", "torch==2.1.2")
)
stunt_image = add_experiment(stunt_image, "stunt")

gftab_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy<2", "pandas", "scikit-learn==1.3.2", "torch==2.1.2",
        # XGBoost 2.x supports device="cuda" with tree_method="hist".
        "xgboost==2.1.4", "einops==0.7.0", "scipy", "matplotlib",
    )
)
gftab_image = add_experiment(gftab_image, "gftab")

rankup_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy<2", "pandas", "scikit-learn==1.3.2", "scipy",
        "torch==2.1.2", "tabm", "optuna==3.6.1",
    )
)
rankup_image = add_experiment(rankup_image, "rankup")
rankup_classification_image = add_experiment(rankup_image, "rankup_classification")


@app.function(
    image=sslae_image,
    gpu="T4",
    timeout=60 * 60 * 4,
    volumes={"/mnt/task3-data": data_volume, "/mnt/task3-results": results_volume},
)
def run_sslae(dataset: str, epochs: int = 100, seed: int = 42):
    import subprocess
    import sys

    subprocess.run([
        sys.executable, "-m", "experiments.sslae.run", "--dataset", dataset,
        "--data-root", "/mnt/task3-data", "--output-root", "/mnt/task3-results",
        "--epochs", str(epochs), "--seed", str(seed),
    ], check=True, cwd="/opt/task3")
    results_volume.commit()


@app.function(
    image=prepare_image,
    timeout=60 * 30,
    volumes={"/mnt/task3-data": data_volume},
)
def prepare_dataset(dataset: str):
    import subprocess
    import sys

    subprocess.run([
        sys.executable, "/opt/task3/prepare_data.py", "--dataset", dataset,
        "--config", "/opt/task3/config/datasets.json", "--output", "/mnt/task3-data",
    ], check=True)
    data_volume.commit()


@app.function(
    image=vime_image,
    gpu="T4",
    timeout=60 * 60 * 4,
    volumes={"/mnt/task3-data": data_volume, "/mnt/task3-results": results_volume},
)
def run_vime(dataset: str, iterations: int = 1000, seed: int = 42):
    import subprocess
    import sys

    subprocess.run([
        sys.executable, "-m", "experiments.vime.run", "--dataset", dataset,
        "--data-root", "/mnt/task3-data", "--output-root", "/mnt/task3-results",
        "--iterations", str(iterations), "--seed", str(seed),
    ], check=True, cwd="/opt/task3")
    results_volume.commit()


@app.function(
    image=stunt_image,
    gpu="T4",
    timeout=60 * 60 * 4,
    volumes={"/mnt/task3-data": data_volume, "/mnt/task3-results": results_volume},
)
def run_stunt(dataset: str, steps: int = 1000, seed: int = 42):
    import subprocess
    import sys

    subprocess.run([
        sys.executable, "-m", "experiments.stunt.run", "--dataset", dataset,
        "--data-root", "/mnt/task3-data", "--output-root", "/mnt/task3-results",
        "--steps", str(steps), "--seed", str(seed),
    ], check=True, cwd="/opt/task3")
    results_volume.commit()


@app.function(
    image=gftab_image,
    gpu="T4",
    timeout=60 * 60 * 4,
    volumes={"/mnt/task3-data": data_volume, "/mnt/task3-results": results_volume},
)
def run_gftab(dataset: str, epochs: int = 10, seed: int = 42):
    import subprocess
    import sys

    subprocess.run([
        sys.executable, "-m", "experiments.gftab.run", "--dataset", dataset,
        "--data-root", "/mnt/task3-data", "--output-root", "/mnt/task3-results",
        "--epochs", str(epochs), "--seed", str(seed),
    ], check=True, cwd="/opt/task3")
    results_volume.commit()


@app.function(
    image=rankup_image,
    gpu="T4",
    timeout=60 * 60 * 4,
    volumes={"/mnt/task3-data": data_volume, "/mnt/task3-results": results_volume},
)
def run_rankup(
    dataset: str, augmentation: str, epochs: int, optuna_epochs: int,
    n_trials: int, seed: int,
):
    import subprocess
    import sys

    subprocess.run([
        sys.executable, "-m", f"experiments.rankup.{augmentation}", "--dataset", dataset,
        "--data-root", "/mnt/task3-data",
        "--output-root", "/mnt/task3-results", "--epochs", str(epochs),
        "--optuna-epochs", str(optuna_epochs), "--n-trials", str(n_trials),
        "--seed", str(seed),
    ], check=True, cwd="/opt/task3")
    results_volume.commit()


@app.function(
    image=rankup_classification_image,
    gpu="T4",
    timeout=60 * 60 * 4,
    volumes={"/mnt/task3-data": data_volume, "/mnt/task3-results": results_volume},
)
def run_rankup_classification(
    dataset: str, augmentation: str, epochs: int, optuna_epochs: int,
    n_trials: int, seed: int,
):
    import subprocess
    import sys

    subprocess.run([
        sys.executable, "-m", f"experiments.rankup_classification.{augmentation}",
        "--dataset", dataset, "--data-root", "/mnt/task3-data",
        "--output-root", "/mnt/task3-results", "--epochs", str(epochs),
        "--optuna-epochs", str(optuna_epochs), "--n-trials", str(n_trials),
        "--seed", str(seed),
    ], check=True, cwd="/opt/task3")
    results_volume.commit()


@app.local_entrypoint()
def main(
    dataset: str, action: str = "prepare", epochs: int = 100, seed: int = 42,
    stunt_steps: int = 1000, gftab_epochs: int = 10,
    augmentation: str = "all", rankup_epochs: int = 120,
    rankup_optuna_epochs: int = 40, rankup_trials: int = 40,
    max_parallel_jobs: int = 4,
):
    actions = {"prepare", "sslae", "vime", "stunt", "gftab", "rankup", "rankup_cls", "all"}
    if action not in actions:
        raise ValueError(f"action must be one of {sorted(actions)}")
    if max_parallel_jobs < 1:
        raise ValueError("--max-parallel-jobs must be at least 1")

    config_path = Path(__file__).parent / "config/datasets.json"
    configured = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_types = {entry["name"]: entry.get("task_type", "classification")
                     for entry in configured["datasets"]}
    available = list(dataset_types)
    if dataset in {"all", "ordinal", "classification"}:
        selected_type = {
            "ordinal": "ordinal_regression",
            "classification": "classification",
        }.get(dataset)
        if dataset == "all" and action == "rankup":
            selected_type = "ordinal_regression"
        elif dataset == "all" and action == "rankup_cls":
            selected_type = "classification"
        selected = [
            name for name, task_type in dataset_types.items()
            if selected_type is None or task_type == selected_type
        ]
    else:
        selected = [dataset]
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(
            f"unknown dataset(s): {unknown}; choose one of {available}, 'ordinal', 'classification', or 'all'"
        )

    incompatible = [
        name for name in selected
        if action == "rankup" and dataset_types[name] != "ordinal_regression"
        or action == "rankup_cls" and dataset_types[name] != "classification"
    ]
    if incompatible:
        raise ValueError(f"Dataset type is incompatible with {action}: {incompatible}")

    rankup_methods = (
        "tabcutmix", "scarf", "range_limited", "gaussian", "zero_out", "tabcutmix_plus"
    )
    if action in {"rankup", "rankup_cls", "all"}:
        methods = rankup_methods if augmentation == "all" else (augmentation,)
        invalid = sorted(set(methods) - set(rankup_methods))
        if invalid:
            raise ValueError(f"unknown RankUp augmentation(s): {invalid}; choose one of {rankup_methods} or 'all'")

    if action == "prepare":
        for name in selected:
            print(f"Preparing dataset={name}")
            prepare_dataset.remote(name)
        return

    tasks = []
    for name in selected:
        task_actions = (
            ("sslae", "vime", "stunt", "gftab", "rankup", "rankup_cls")
            if action == "all" else (action,)
        )
        for task_action in task_actions:
            if task_action in {"sslae", "vime", "stunt", "gftab"}:
                runner = {
                    "sslae": run_sslae, "vime": run_vime,
                    "stunt": run_stunt, "gftab": run_gftab,
                }[task_action]
                if task_action == "sslae":
                    tasks.append((runner, (name, epochs, seed), {}))
                elif task_action == "vime":
                    tasks.append((runner, (name,), {"seed": seed}))
                elif task_action == "stunt":
                    tasks.append((runner, (name,), {"steps": stunt_steps, "seed": seed}))
                else:
                    tasks.append((runner, (name,), {"epochs": gftab_epochs, "seed": seed}))
            elif task_action == "rankup" and dataset_types[name] == "ordinal_regression":
                for method in methods:
                    trials = rankup_trials or 40
                    tasks.append((run_rankup, (name, method, rankup_epochs,
                                               rankup_optuna_epochs, trials, seed), {}))
            elif task_action == "rankup_cls" and dataset_types[name] == "classification":
                for method in methods:
                    trials = rankup_trials or 40
                    tasks.append((run_rankup_classification, (name, method, rankup_epochs,
                                                               rankup_optuna_epochs, trials, seed), {}))

    print(f"Submitting {len(tasks)} training jobs, up to {max_parallel_jobs} concurrently")
    for start in range(0, len(tasks), max_parallel_jobs):
        batch = tasks[start:start + max_parallel_jobs]
        calls = [runner.spawn(*args, **kwargs) for runner, args, kwargs in batch]
        modal.FunctionCall.gather(*calls)
