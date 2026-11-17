from pathlib import Path
import csv
import time

from stable_baselines3 import PPO
from minigrid.wrappers import FlatObsWrapper
from src.environments.custom_empty import VariableSizeEmptyEnv


def make_env(size):
    env = VariableSizeEmptyEnv(size=size)
    env = FlatObsWrapper(env)
    return env


def evaluate_model(model_path, test_size, eval_seeds):
    model = PPO.load(model_path)
    env = make_env(test_size)
    successes = 0
    timeouts = 0
    total_reward = 0
    total_steps = 0
    start_time = time.time()
    for eval_seed in eval_seeds:
        obs, info = env.reset(seed=eval_seed)
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        total_reward += episode_reward
        total_steps += steps
        success = episode_reward > 0
        if success:
            successes += 1
        elif truncated:
            timeouts += 1

    evaluation_time = time.time() - start_time
    env.close()

    return {
        "success_rate": successes / len(eval_seeds),
        "average_reward": total_reward / len(eval_seeds),
        "average_steps": total_steps / len(eval_seeds),
        "timeout_rate": timeouts / len(eval_seeds),
        "evaluation_time_seconds": evaluation_time,
    }


def save_result(
    algorithm,
    model_name,
    train_size,
    test_size,
    timesteps,
    seed,
    episodes,
    results_dict,
):
    output_file = Path("results") / "evaluations" / "evaluation_results.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_file.exists()

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "algorithm",
                "model_name",
                "train_size",
                "test_size",
                "timesteps",
                "seed",
                "episodes",
                "success_rate",
                "timeout_rate",
                "average_reward",
                "average_steps",
                "evaluation_time_seconds"
            ])

        writer.writerow([
            algorithm,
            model_name,
            train_size,
            test_size,
            timesteps,
            seed,
            episodes,
            results_dict["success_rate"],
            results_dict["timeout_rate"],
            results_dict["average_reward"],
            results_dict["average_steps"],
            results_dict["evaluation_time_seconds"]
        ])


if __name__ == "__main__":
    # CONFIG

    SEEDS = [42, 123, 999]
    TEST_SIZES = [5, 10, 15, 20, 30, 40]

    # ensures every evaluation is done with the same seeds for consistency
    # reduces chance of "bad" environments that are too easy or hard,
    # ensuring fairness across models
    EVAL_SEEDS = list(range(1000, 1250))

    # IMPORTANT
    # UNCOMMENT BLOCK BELOW TO EVALUATE DIFFERENT TRAINING CONFIGURATIONS

    # Original baseline PPO 250k, trained on 5x5, 10x10, 15x15, and 20x20:
    # TRAIN_SIZES = [5, 10, 15, 20]
    # TIMESTEPS = 250_000
    # MODEL_FOLDER = "baseline"
    # MODEL_NAME_TEMPLATE = (
    #     "ppo_{train_size}x{train_size}_baseline_250k_seed{seed}"
    # )
    # ALGORITHM = "PPO"

    # Mixed PPO 250k
    # TRAIN_SIZES = ["mixed"]  # mixed
    # TIMESTEPS = 250_000  # mixed 250k
    # MODEL_FOLDER = "mixed"  # mixed
    # MODEL_NAME_TEMPLATE = "ppo_mixed_250k_seed{seed}"  # mixed
    # ALGORITHM = "PPO_MIXED"  # mixed

    # Mixed Optuna PPO 250k:
    # TRAIN_SIZES = ["mixed"]
    # TIMESTEPS = 250_000
    # MODEL_FOLDER = "mixed_optuna"
    # MODEL_NAME_TEMPLATE = "ppo_mixed_optuna_250k_seed{seed}"
    # ALGORITHM = "PPO_MIXED_OPTUNA"

    # 500k fixed-size comparison model:
    # TRAIN_SIZES = [15]
    # TIMESTEPS = 500_000
    # MODEL_FOLDER = "baseline"
    # MODEL_NAME_TEMPLATE = (
    #     "ppo_{train_size}x{train_size}_baseline_500k_seed{seed}"
    # )
    # ALGORITHM = "PPO"

    # 250k Optuna-tuned fixed 15x15 model (active):
    TRAIN_SIZES = [15]
    TIMESTEPS = 250_000
    MODEL_FOLDER = "baseline_optuna"
    MODEL_NAME_TEMPLATE = "ppo_15x15_optuna_250k_seed{seed}"
    ALGORITHM = "PPO_OPTUNA"

    # To evaluate a PLR model, uncomment exactly one block matching the
    # model's training budget:

    # PLR-inspired PPO 50k validation model:
    # TRAIN_SIZES = ["plr"]
    # TIMESTEPS = 50_000
    # MODEL_FOLDER = "plr_inspired"
    # MODEL_NAME_TEMPLATE = "ppo_plr_inspired_50k_seed{seed}"
    # ALGORITHM = "PPO_PLR_INSPIRED"

    # PLR-inspired PPO 250k:
    # TRAIN_SIZES = ["plr"]
    # TIMESTEPS = 250_000
    # MODEL_FOLDER = "plr_inspired"
    # MODEL_NAME_TEMPLATE = "ppo_plr_inspired_250k_seed{seed}"
    # ALGORITHM = "PPO_PLR_INSPIRED"

    # PLR-inspired PPO 500k:
    # TRAIN_SIZES = ["plr"]
    # TIMESTEPS = 500_000
    # MODEL_FOLDER = "plr_inspired"
    # MODEL_NAME_TEMPLATE = "ppo_plr_inspired_500k_seed{seed}"
    # ALGORITHM = "PPO_PLR_INSPIRED"

    # 500k Optuna mixed-size comparison model (active):
    # TRAIN_SIZES = ["mixed"]  # mixed_optuna
    # TIMESTEPS = 500_000  # mixed_optuna 500k
    # MODEL_FOLDER = "mixed_optuna"  # mixed_optuna
    # MODEL_NAME_TEMPLATE = "ppo_mixed_optuna_500k_seed{seed}"  # mixed_optuna
    # ALGORITHM = "PPO_MIXED_OPTUNA"  # mixed_optuna

    for train_size in TRAIN_SIZES:
        for seed in SEEDS:
            model_name = MODEL_NAME_TEMPLATE.format(
                train_size=train_size,
                seed=seed,
            )
            model_path = Path("saved models") / MODEL_FOLDER / model_name

            if not model_path.with_suffix(".zip").exists():
                print(f"Skipping missing model: {model_name}")
                continue

            print(f"Evaluating: {model_name}")

            for test_size in TEST_SIZES:
                results = evaluate_model(
                    model_path=str(model_path),
                    test_size=test_size,
                    eval_seeds=EVAL_SEEDS,
                )

                save_result(
                    algorithm=ALGORITHM,
                    model_name=model_name,
                    train_size=train_size,
                    test_size=test_size,
                    timesteps=TIMESTEPS,
                    seed=seed,
                    episodes=len(EVAL_SEEDS),
                    results_dict=results,
                )

                print(
                    f"Trained on: {train_size}x{train_size}"
                    f" -> Test {test_size}x{test_size}"
                    f" | Success = "
                    f"{results['success_rate']:.3f}"
                    f" | Timeout = "
                    f"{results['timeout_rate']:.3f}"
                )

    print("\nALL EVALUATIONS COMPLETE")