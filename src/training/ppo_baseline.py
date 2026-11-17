from pathlib import Path
import json
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from minigrid.wrappers import FlatObsWrapper

from src.environments.custom_empty import VariableSizeEmptyEnv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_env(size: int):
    """
    Create environment and apply wrappers.
    """
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
    """
    Train a PPO model and save:

    - Model ZIP
    - Metadata JSON

    """
    env = make_env(size)
    tensorboard_path = PROJECT_ROOT / "logs" / "tensorboard"
    tensorboard_path.mkdir(parents=True, exist_ok=True)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        verbose=1,
        tensorboard_log=str(tensorboard_path),
    )

    print(f"TRAINING PPO ON {size}x{size}")
    print(f"TIMESTEPS: {timesteps}")

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

    # SAVE METADATA
    metadata_path = (
        PROJECT_ROOT / "results" / "training" / f"{model_name}.json"
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "algorithm": "PPO",
        "environment": "VariableSizeEmptyEnv",
        "map_size": size,
        "timesteps": timesteps,
        "model_name": model_name,
        "model_path": model_path.relative_to(PROJECT_ROOT).as_posix(),
        "metadata_path": metadata_path.relative_to(PROJECT_ROOT).as_posix(),
        "save_folder": save_folder,
        "seed": seed,
        "tensorboard_run": tensorboard_log_dir.name,
        "tensorboard_log_dir": (
            tensorboard_log_dir.relative_to(PROJECT_ROOT).as_posix()
        ),
        "policy": "MlpPolicy",
        "wrapper": "FlatObsWrapper",
        "training_time_seconds": elapsed_seconds,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nMetadata saved:")
    print(metadata_path)

    env.close()


if __name__ == "__main__":
    # Final 250k fixed-size baseline models:
    TRAIN_SIZES = [5, 10, 15, 20]
    TIMESTEPS = 250_000

    # To reproduce the saved 500k fixed 15x15 models, comment out the active
    # settings above and uncomment these:
    # TRAIN_SIZES = [15]
    # TIMESTEPS = 500_000

    SEED = [42, 123, 999]
    SAVE_FOLDER = "baseline"

    for size in TRAIN_SIZES:
        for seed in SEED:
            model_name = (
                f"ppo_{size}x{size}_baseline_"
                f"{TIMESTEPS // 1000}k_seed{seed}"
            )

            train_ppo(
                size=size,
                timesteps=TIMESTEPS,
                model_name=model_name,
                save_folder=SAVE_FOLDER,
                seed=seed,
            )

    print("\nALL BASELINE TRAINING COMPLETE")