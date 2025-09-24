import copy
import json
import os
import sys
from argparse import ArgumentParser
from graphlib import CycleError, TopologicalSorter
from typing import Any, Callable, Optional

import dill
import optuna

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
)  # needed for the environment
from src.hyperparameter_search.search_space.optuna_search_space import (
    Dependency,
    OptunaFloat,
    OptunaInt,
    OptunaSearchSpace,
)
from src.hyperparameter_search.utils import (
    find_search_spaces,
    get_value_at_path,
    in_path,
    set_value_at_path,
)
from src.main import run_with_optuna
from src.parameters import Parameters


def get_real_dependency_path(config: dict, dependency: str):
    """
    Resolves the dependency path in the configuration.
    Useful for cases where the value depended on is inside a categorical search space.
    If it turns that this is not the case, the code will throw an error later.

    If the dependency is not found, it will raise a ValueError.
    """
    while not in_path(config, dependency):
        if "." not in dependency:
            raise ValueError(
                f"Dependency '{dependency}' not found in the configuration."
            )
        dependency = ".".join(dependency.split(".")[:-1])
    return dependency


def get_dependency_value(config: dict, dependency: str | dict | int | float):
    """
    Gets the value at the dependency path in the configuration.
    If a transformation is provided, it applies it to the value.
    """
    if isinstance(dependency, dict):
        dependency: Dependency[int | float] = dependency
        value = get_value_at_path(config, dependency["path"])
        if "transformation" in dependency:
            value = dependency["transformation"](value)
    elif isinstance(dependency, str):
        value = get_value_at_path(config, dependency)
    elif isinstance(dependency, int) or isinstance(dependency, float):
        value = dependency
    else:
        raise TypeError(
            f"Dependency must be a string or a dict, got {type(dependency).__name__}."
        )
    return value


def get_dependencies_from_int_or_float(
    config: dict,
    args: OptunaInt | OptunaFloat,
):
    dependencies = set()
    if isinstance(args["min"], str):
        dependency = get_real_dependency_path(config, args["min"])
        dependencies.add(dependency)
    elif isinstance(args["min"], dict):
        dependency = get_real_dependency_path(config, args["min"]["path"])
        dependencies.add(dependency)
    if isinstance(args["max"], str):
        dependency = get_real_dependency_path(config, args["max"])
        dependencies.add(dependency)
    elif isinstance(args["max"], dict):
        dependency = get_real_dependency_path(config, args["max"]["path"])
        dependencies.add(dependency)
    return dependencies


def generate_optuna_configuration(
    base_config: dict, trial: optuna.trial.Trial
) -> tuple[str, dict]:
    """
    Generates a named configuration for a single Optuna trial based on the base config.
    """
    search_spaces_with_paths = find_search_spaces(base_config)

    if not search_spaces_with_paths:
        return "base", base_config

    paths, search_space_objects = zip(*search_spaces_with_paths)
    search_space_objects: list[OptunaSearchSpace]

    new_config = copy.deepcopy(base_config)
    name_parts = []

    dependencies: dict[str, set[str]] = {}
    lookup: dict[str, OptunaSearchSpace] = {}

    for path, space_obj in zip(paths, search_space_objects):
        key = ".".join(map(str, path))
        dependencies[key] = set()
        lookup[key] = space_obj
        if space_obj.type == "int" or space_obj.type == "float":
            dependencies[key].update(
                get_dependencies_from_int_or_float(new_config, space_obj.args)
            )
        elif space_obj.type == "list":
            # num_entries is always an OptunaInt
            dependencies[key].update(
                get_dependencies_from_int_or_float(
                    new_config, space_obj.args["num_entries"]
                )
            )

            # entry can be an Categorical, which does not support dependencies
            if space_obj.args["entry"].get("choices") is None:
                dependencies[key].update(
                    get_dependencies_from_int_or_float(
                        new_config, space_obj.args["entry"]
                    )
                )

    ts = TopologicalSorter(dependencies)
    sorted_order = tuple(ts.static_order())

    # Sort the search spaces and paths based on the topological order
    sorted_spaces = [lookup[key] for key in sorted_order if key in lookup]
    sorted_paths = [path.split(".") for path in sorted_order if path in lookup]

    for path_list, space_obj in zip(sorted_paths, sorted_spaces):
        if space_obj.type == "int":
            min_value = get_dependency_value(new_config, space_obj.args["min"])
            max_value = get_dependency_value(new_config, space_obj.args["max"])
            value = trial.suggest_int(
                space_obj.label,
                min_value,
                max_value,
                step=space_obj.args.get("step", 1),
                log=space_obj.args.get("log", False),
            )
        elif space_obj.type == "float":
            min_value = get_dependency_value(new_config, space_obj.args["min"])
            max_value = get_dependency_value(new_config, space_obj.args["max"])
            value = trial.suggest_float(
                space_obj.label,
                min_value,
                max_value,
                step=space_obj.args.get("step", 1),
                log=space_obj.args.get("log", False),
            )
        elif space_obj.type == "list":
            min_entries = get_dependency_value(
                new_config, space_obj.args["num_entries"]["min"]
            )
            max_entries = get_dependency_value(
                new_config, space_obj.args["num_entries"]["max"]
            )
            num_entries = trial.suggest_int(
                f"{space_obj.label}_num",
                min_entries,
                max_entries,
                step=space_obj.args["num_entries"].get("step", 1),
                log=space_obj.args["num_entries"].get("log", False),
            )
            values = []
            for i in range(num_entries):
                entry_label = f"{space_obj.label}_entry_{i}"
                if space_obj.args["entry"].get("choices") is None:
                    min_value = get_dependency_value(
                        new_config, space_obj.args["entry"]["min"]
                    )
                    max_value = get_dependency_value(
                        new_config, space_obj.args["entry"]["max"]
                    )
                    if isinstance(min_value, int):
                        value = trial.suggest_int(
                            entry_label,
                            min_value,
                            max_value,
                            step=space_obj.args["entry"].get("step", 1),
                            log=space_obj.args["entry"].get("log", False),
                        )
                    else:
                        value = trial.suggest_float(
                            entry_label,
                            min_value,
                            max_value,
                            step=space_obj.args["entry"].get("step", 1),
                            log=space_obj.args["entry"].get("log", False),
                        )
                else:
                    value = trial.suggest_categorical(
                        entry_label, space_obj.args["entry"]["choices"]
                    )
                values.append(value)
        elif space_obj.type == "categorical":
            value = trial.suggest_categorical(
                space_obj.label, space_obj.args["choices"]
            )
        else:
            raise ValueError(f"Unsupported Optuna search space type: {space_obj.type}")

        set_value_at_path(new_config, path_list, value)

        if space_obj.label:
            formatted_value = space_obj.value_formatter(value)
            name_parts.append(f"{space_obj.label}={formatted_value}")

    experiment_name = "params_" + ("_".join(name_parts) or "base")

    return experiment_name, new_config


class Objective:
    def __init__(
        self,
        parameters: dict,
        results_dir: str,
        experiment_name: str,
        use_torch_compile: bool,
        gpu_id: Optional[int],
        loss_term_scaling: Optional[dict[str, float]],
    ):
        self.parameters = parameters
        self.results_dir = results_dir
        self.experiment_name = experiment_name
        self.gpu_id = gpu_id
        self.use_torch_compile = use_torch_compile
        self.loss_term_scaling = loss_term_scaling

    def __call__(self, trial: optuna.trial.Trial) -> float:
        name, config = generate_optuna_configuration(self.parameters, trial)
        run = 1
        if os.path.exists(os.path.join(self.results_dir, self.experiment_name, name)):
            # If the directory already exists, increment the run number
            existing_runs = [
                int(d)
                for d in os.listdir(
                    os.path.join(self.results_dir, self.experiment_name, name)
                )
                if d.isdigit()
            ]
            run = max(existing_runs) + 1 if existing_runs else 1

        config = {
            **copy.deepcopy(Parameters.default()),
            **config,
            "run_directory": os.path.join(
                self.results_dir, self.experiment_name, name, f"{run:03d}"
            ),
            "experiment": self.experiment_name,
            "name": name,
            "run": run,
        }

        os.makedirs(config["run_directory"], exist_ok=True)
        # Write the configuration to a JSON file
        with open(os.path.join(config["run_directory"], "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        return run_with_optuna(
            config,
            trial,
            self.loss_term_scaling,
            self.gpu_id,
            self.use_torch_compile,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the base configuration pickle file.",
        required=True,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save results.",
        required=True,
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="optuna_experiment",
        help="Name of the experiment.",
        required=True,
    )
    parser.add_argument(
        "--use_torch_compile",
        action="store_true",
        help="Whether to use torch.compile for optimization.",
        default=False,
    )
    parser.add_argument(
        "--study_name", type=str, help="Name of the Optuna study.", required=True
    )
    parser.add_argument(
        "--storage",
        type=str,
        help="Name of the sqlite database for Optuna storage. It is assumed to be in the experiment directory.",
        required=True,
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU ID to use for training. If not given, no GPU is used.",
    )
    parser.add_argument(
        "--num_trials", type=int, required=True, help="Number of trials to run."
    )
    parser.add_argument(
        "--optuna_loss_scaling_path",
        type=str,
        default=None,
        help="Path to the Optuna loss scaling configuration file.",
    )

    args = parser.parse_args()
    config = dill.load(open(args.config, "rb"))
    loss_scaling = None
    if args.optuna_loss_scaling_path is not None:
        loss_scaling = dill.load(open(args.optuna_loss_scaling_path, "rb"))

    study = optuna.load_study(
        study_name=args.study_name,
        storage=f"sqlite:///{os.path.join(args.results_dir, args.experiment_name, args.storage)}",
    )
    study.optimize(
        Objective(
            parameters=config,
            results_dir=args.results_dir,
            experiment_name=args.experiment_name,
            use_torch_compile=args.use_torch_compile,
            gpu_id=args.gpu_id,
            loss_term_scaling=loss_scaling,
        ),
        n_trials=args.num_trials,
    )
