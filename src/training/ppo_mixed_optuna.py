from pathlib import Path

import json
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.environments.mixed_size import MixedSizeEnv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


TRAIN_SIZES = [5, 10, 15, 20]


def make_env(seed):
    env = MixedSizeEnv(sizes=TRAIN_SIZES, seed=seed)
    env = Monitor(env)
    return env


def train_mixed_ppo(
    timesteps: int,
    model_name: str,
    save_folder: str,
    seed: int,
):
    env = make_env(seed)
    tensorboard_path = PROJECT_ROOT / "logs" / "tensorboard"
    tensorboard_path.mkdir(parents=True, exist_ok=True)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        verbose=1,
        tensorboard_log=str(tensorboard_path),
        learning_rate=3.8857991779259546e-05,
        n_steps=2048,
        batch_size=32,
        ent_coef=0.008834149776727012,
        clip_range=0.19125610718247976,
    )

    print("TRAINING MIXED PPO")
    print(f"Training Sizes: {TRAIN_SIZES}")
    print("Selection Method: random_uniform")
    print(f"Seed: {seed}")
    print(f"Timesteps: {timesteps}")

    start_time = time.time()
    model.learn(total_timesteps=timesteps, tb_log_name=model_name)
    elapsed_seconds = time.time() - start_time
    minutes = int(elapsed_seconds // 60)
    seconds = elapsed_seconds % 60
    print(f"\nTraining Time: {minutes} min {seconds:.1f} sec")

    # SAVE MODEL
    save_path = PROJECT_ROOT / "saved models" / save_folder / model_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    model_path = save_path.with_suffix(".zip")
    tensorboard_log_dir = Path(model.logger.dir)

    print("\nModel saved:")
    print(model_path)
    print(f"TensorBoard log: {tensorboard_log_dir}")

    # BUILD TRAINING STATISTICS
    counts = env.unwrapped.get_size_counts()
    total_episodes = sum(counts.values())
    actual_distribution = {
        str(size): (
            counts[size] / total_episodes
            if total_episodes > 0
            else 0
        )
        for size in counts
    }

    # METADATA
    metadata_path = (
        PROJECT_ROOT / "results" / "training" / f"{model_name}.json"
    )
    metadata = {
        "algorithm": "PPO_MIXED_OPTUNA",
        "environment": "MixedSizeEnv",
        "training_sizes": TRAIN_SIZES,
        "selection_method": "random_uniform",
        "seed": seed,
        "timesteps": timesteps,
        "policy": "MlpPolicy",
        "hyperparameters": {
            "learning_rate": 3.8857991779259546e-05,
            "n_steps": 2048,
            "batch_size": 32,
            "ent_coef": 0.008834149776727012,
            "clip_range": 0.19125610718247976,
        },
        "wrappers": ["FlatObsWrapper", "Monitor"],
        "model_name": model_name,
        "model_path": model_path.relative_to(PROJECT_ROOT).as_posix(),
        "metadata_path": metadata_path.relative_to(PROJECT_ROOT).as_posix(),
        "tensorboard_run": tensorboard_log_dir.name,
        "tensorboard_log_dir": (
            tensorboard_log_dir.relative_to(PROJECT_ROOT).as_posix()
        ),
        "save_folder": save_folder,
        "training_time_seconds": elapsed_seconds,
        "selection_distribution_expected": {
            "5": 0.25,
            "10": 0.25,
            "15": 0.25,
            "20": 0.25,
        },
        "selection_distribution_actual": actual_distribution,
        "size_counts": counts,
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nMetadata saved:")
    print(metadata_path)
    print("\nMap-size selection counts:")
    print(metadata["size_counts"])
    print("\nActual sampling distribution:")
    print(metadata["selection_distribution_actual"])
    env.close()


if __name__ == "__main__":
    TIMESTEPS = 250_000
    # To run 500k instead, comment out the line above and uncomment below:
    # TIMESTEPS = 500_000
    SAVE_FOLDER = "mixed_optuna"
    SEEDS = [42, 123, 999]

    # Model names use the matching timestep suffix automatically.
    for seed in SEEDS:
        train_mixed_ppo(
            timesteps=TIMESTEPS,
            model_name=(
                f"ppo_mixed_optuna_{TIMESTEPS // 1000}k_seed{seed}"
            ),
            save_folder=SAVE_FOLDER,
            seed=seed,
        )

    print("\nALL MIXED PPO TRAINING COMPLETE")
    print("\nOPTUNA-TUNED PPO PARAMETERS")
    print(f"learning_rate={3.8857991779259546e-05}")
    print(f"n_steps={2048}")
    print(f"batch_size={32}")
    print(f"ent_coef={0.008834149776727012}")
    print(f"clip_range={0.19125610718247976}")
