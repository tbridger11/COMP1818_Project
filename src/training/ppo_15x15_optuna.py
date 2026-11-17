from pathlib import Path
import json
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from minigrid.wrappers import FlatObsWrapper

from src.environments.custom_empty import VariableSizeEmptyEnv


OPTUNA_PARAMS = {
    "learning_rate": 3.8857991779259546e-05,
    "n_steps": 2048,
    "batch_size": 32,
    "ent_coef": 0.008834149776727012,
    "clip_range": 0.19125610718247976,
}


def make_env(size: int):
    env = VariableSizeEmptyEnv(size=size)
    env = FlatObsWrapper(env)
    env = Monitor(env)
    return env


def train_ppo(
    size: int,
    timesteps: int,
    model_name: str,
    save_folder: str,
    seed: int,
):
    env = make_env(size)
    Path("logs/tensorboard").mkdir(parents=True, exist_ok=True)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        verbose=1,
        tensorboard_log="logs/tensorboard",
        **OPTUNA_PARAMS,
    )

    print(f"TRAINING OPTUNA-TUNED PPO ON {size}x{size}")
    print(f"TIMESTEPS: {timesteps}")
    print(f"OPTUNA HYPERPARAMETERS: {OPTUNA_PARAMS}")

    start_time = time.time()
    model.learn(total_timesteps=timesteps, tb_log_name=model_name)
    elapsed_seconds = time.time() - start_time

    minutes = int(elapsed_seconds // 60)
    seconds = elapsed_seconds % 60

    print(f"\nTraining Time: {minutes} min {seconds:.1f} sec")

    save_path = Path("saved models") / save_folder / model_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    model_path = save_path.with_suffix(".zip")
    tensorboard_log_dir = Path(model.logger.dir)

    print("\nModel saved:")
    print(model_path)
    print(f"TensorBoard log: {tensorboard_log_dir}")

    metadata_path = Path("results") / "training" / f"{model_name}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "algorithm": "PPO_OPTUNA",
        "environment": "VariableSizeEmptyEnv",
        "map_size": size,
        "timesteps": timesteps,
        "model_name": model_name,
        "model_path": model_path.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "save_folder": save_folder,
        "seed": seed,
        "tensorboard_run": tensorboard_log_dir.name,
        "tensorboard_log_dir": tensorboard_log_dir.as_posix(),
        "policy": "MlpPolicy",
        "wrappers": ["FlatObsWrapper", "Monitor"],
        "hyperparameters": OPTUNA_PARAMS,
        "training_time_seconds": elapsed_seconds,
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nMetadata saved:")
    print(metadata_path)

    env.close()


if __name__ == "__main__":
    TRAIN_SIZE = 15
    TIMESTEPS = 250_000
    # To run 500k instead, comment out the line above and uncomment below:
    # TIMESTEPS = 500_000
    SEEDS = [42, 123, 999]
    SAVE_FOLDER = "baseline_optuna"

    # Model names use the matching timestep suffix automatically.
    for seed in SEEDS:
        model_name = (
            f"ppo_{TRAIN_SIZE}x{TRAIN_SIZE}_optuna_"
            f"{TIMESTEPS // 1000}k_seed{seed}"
        )

        train_ppo(
            size=TRAIN_SIZE,
            timesteps=TIMESTEPS,
            model_name=model_name,
            save_folder=SAVE_FOLDER,
            seed=seed,
        )

    print("\nALL 15x15 OPTUNA PPO TRAINING COMPLETE")
