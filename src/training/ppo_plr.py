from pathlib import Path

import json
import math
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.environments.plr import (
    MAX_PRIORITY,
    MIN_PRIORITY,
    PRIORITY_FAILURE_MULTIPLIER,
    PRIORITY_SUCCESS_MULTIPLIER,
    UNIFORM_SAMPLING_PROBABILITY,
    PLREnv,
)


TRAIN_SIZES = [5, 10, 15, 20]
# Best parameters obtained from src/training/tune_mixed_ppo.py.
OPTUNA_PARAMS = {
    "learning_rate": 3.8857991779259546e-05,
    "n_steps": 2048,
    "batch_size": 32,
    "ent_coef": 0.008834149776727012,
    "clip_range": 0.19125610718247976,
}


def make_env(seed):
    env = PLREnv(sizes=TRAIN_SIZES, seed=seed)
    env = Monitor(env)
    return env


def train_ppo_plr(
    timesteps: int,
    model_name: str,
    save_folder: str,
    seed: int,
):
    env = make_env(seed)
    Path("logs/tensorboard").mkdir(parents=True, exist_ok=True)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        verbose=1,
        tensorboard_log="logs/tensorboard",
        learning_rate=OPTUNA_PARAMS["learning_rate"],
        n_steps=OPTUNA_PARAMS["n_steps"],
        batch_size=OPTUNA_PARAMS["batch_size"],
        ent_coef=OPTUNA_PARAMS["ent_coef"],
        clip_range=OPTUNA_PARAMS["clip_range"],
    )

    print("TRAINING PPO WITH PLR-INSPIRED SIZE SELECTION")
    print(f"Training Sizes: {TRAIN_SIZES}")
    print(f"Seed: {seed}")
    print(f"Timesteps: {timesteps}")
    print(f"Optuna hyperparameters: {OPTUNA_PARAMS}")

    start_time = time.time()
    model.learn(total_timesteps=timesteps, tb_log_name=model_name)
    elapsed_seconds = time.time() - start_time
    minutes = int(elapsed_seconds // 60)
    seconds = elapsed_seconds % 60
    print(f"\nTraining Time: {minutes} min {seconds:.1f} sec")

    save_path = Path("saved models") / save_folder / model_name
    model_path = save_path.with_suffix(".zip")
    tensorboard_log_dir = Path(model.logger.dir)

    plr_env = env.unwrapped
    level_stats = plr_env.get_level_stats()
    selection_counts = {
        str(size): stats["times_sampled"]
        for size, stats in level_stats.items()
    }
    successes = {
        str(size): stats["successes"]
        for size, stats in level_stats.items()
    }
    failures = {
        str(size): stats["failures"]
        for size, stats in level_stats.items()
    }
    final_priorities = {
        str(size): stats["priority"]
        for size, stats in level_stats.items()
    }
    selection_distribution_actual = plr_env.get_actual_distribution()

    completed_episodes = sum(successes.values()) + sum(failures.values())
    if completed_episodes != plr_env.completed_episodes:
        raise RuntimeError(
            "Successes and failures do not match completed episodes."
        )

    total_selections = sum(selection_counts.values())
    if total_selections not in (
        completed_episodes,
        completed_episodes + 1,
    ):
        raise RuntimeError(
            "Selection counts must equal completed episodes or exceed them "
            "by one unfinished episode."
        )

    if not math.isclose(
        sum(selection_distribution_actual.values()),
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise RuntimeError("The sampling distribution does not sum to 1.")

    for size, stats in level_stats.items():
        priority = stats["priority"]
        if (
            not math.isfinite(priority)
            or not MIN_PRIORITY <= priority <= MAX_PRIORITY
        ):
            raise RuntimeError(
                f"Priority for size {size} must be finite and between "
                f"{MIN_PRIORITY} and {MAX_PRIORITY}."
            )

    metadata_path = Path("results") / "training" / f"{model_name}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))

    print("\nModel saved:")
    print(model_path)
    print(f"TensorBoard log: {tensorboard_log_dir}")

    metadata = {
        "algorithm": "PPO_PLR_INSPIRED",
        "environment": "PLREnv",
        "curriculum_description": (
            "Select each training size uniformly 20% of the time; otherwise "
            "sample using its current priority."
        ),
        "training_sizes": TRAIN_SIZES,
        "selection_method": "adaptive_priority",
        "uniform_sampling_probability": UNIFORM_SAMPLING_PROBABILITY,
        "success_priority_multiplier": PRIORITY_SUCCESS_MULTIPLIER,
        "failure_priority_multiplier": PRIORITY_FAILURE_MULTIPLIER,
        "minimum_priority": MIN_PRIORITY,
        "maximum_priority": MAX_PRIORITY,
        "seed": seed,
        "timesteps": timesteps,
        "policy": "MlpPolicy",
        "hyperparameters": OPTUNA_PARAMS,
        "wrappers": ["FlatObsWrapper", "Monitor"],
        "model_name": model_name,
        "model_path": model_path.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "tensorboard_run": tensorboard_log_dir.name,
        "tensorboard_log_dir": tensorboard_log_dir.as_posix(),
        "save_folder": save_folder,
        "training_time_seconds": elapsed_seconds,
        "selection_counts": selection_counts,
        "selection_distribution_actual": selection_distribution_actual,
        "successes": successes,
        "failures": failures,
        "final_priorities": final_priorities,
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nMetadata saved:")
    print(metadata_path)
    print("\nSelection counts:")
    print(selection_counts)
    print("\nFinal priorities:")
    print(final_priorities)
    env.close()


if __name__ == "__main__":
    TIMESTEPS = 250_000
    SEEDS = [42, 123, 999]

    # To reproduce the saved 50k validation models, comment out the active
    # settings above and uncomment these:
    # TIMESTEPS = 50_000
    # SEEDS = [42, 123]

    # To reproduce the saved 500k models, instead uncomment these:
    # TIMESTEPS = 500_000
    # SEEDS = [42, 123, 999]

    # Model names use the matching timestep suffix automatically.

    SAVE_FOLDER = "plr_inspired"

    for seed in SEEDS:
        train_ppo_plr(
            timesteps=TIMESTEPS,
            model_name=f"ppo_plr_inspired_{TIMESTEPS // 1000}k_seed{seed}",
            save_folder=SAVE_FOLDER,
            seed=seed,
        )

    print("\nALL PLR-INSPIRED PPO TRAINING COMPLETE")
