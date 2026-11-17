from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import optuna
from stable_baselines3 import PPO

from src.environments.mixed_size import MixedSizeEnv


# CONFIG

TRAIN_SIZES = [5, 10, 15, 20]
EVAL_SIZES = [5, 10, 15, 20]
TRIAL_TIMESTEPS = 250_000
EVAL_EPISODES = 50
N_TRIALS = 40
BASE_SEED = 42
VAL_SEEDS = list(range(5000, 5050))
OUTPUT_PATH = Path("results/optuna_best_params.json")

# EVALUATION

def evaluate_model(model: PPO) -> tuple[float, dict[str, float]]:
    per_size_success_rate = {}

    for size in EVAL_SIZES:
        env = MixedSizeEnv(sizes=[size], seed=BASE_SEED)
        successes = 0

        try:
            for eval_seed in VAL_SEEDS:
                obs, _ = env.reset(seed=eval_seed)

                terminated = False
                truncated = False
                episode_reward = 0

                while not (terminated or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward

                if episode_reward > 0:
                    successes += 1
        finally:
            env.close()

        success_rate = successes / EVAL_EPISODES
        per_size_success_rate[str(size)] = success_rate

    mean_success_rate = float(np.mean(list(per_size_success_rate.values())))
    return mean_success_rate, per_size_success_rate


# OPTUNA OBJECTIVE

def objective(trial: optuna.Trial) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical(
            "n_steps", [128, 256, 512, 1024, 2048]
        ),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.05),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
    }

    print(f"TRIAL {trial.number + 1}/{N_TRIALS}")
    for key, value in params.items():
        print(f"{key}: {value}")

    env = MixedSizeEnv(sizes=TRAIN_SIZES, seed=BASE_SEED)
    model = None

    try:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            seed=BASE_SEED,
            verbose=0,
            **params,
        )
        print(f"\nTraining PPO for {TRIAL_TIMESTEPS:,} timesteps...")
        model.learn(total_timesteps=TRIAL_TIMESTEPS)
        print("Evaluating model...")

        mean_success_rate, per_size_rates = evaluate_model(model)
        trial.set_user_attr("per_size_success_rate", per_size_rates)

        print("\nPer-size success rates:")
        for size in EVAL_SIZES:
            print(f"  {size:>2}x{size:<2}: {per_size_rates[str(size)]:.3f}")

        print(f"\nMean Success Rate: {mean_success_rate:.3f}")
        return mean_success_rate

    finally:
        env.close()
        if model is not None:
            del model
        gc.collect()


def main():
    print(f"Training Sizes: {TRAIN_SIZES}")
    print(f"Evaluation Sizes: {EVAL_SIZES}")
    print(f"Training Timesteps: {TRIAL_TIMESTEPS:,}")
    print(f"Evaluation Episodes: {EVAL_EPISODES}")
    print(f"Trials: {N_TRIALS}")

    sampler = optuna.samplers.TPESampler(seed=BASE_SEED, n_startup_trials=10)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    trials_df = study.trials_dataframe()
    trials_df.to_csv(OUTPUT_PATH.parent / "optuna_all_trials.csv", index=False)
    best_per_size_rates = study.best_trial.user_attrs["per_size_success_rate"]

    result = {
        "best_mean_success_rate": float(study.best_value),
        "best_params": study.best_params,
        "best_per_size_success_rate": best_per_size_rates,
        "best_trial_number": study.best_trial.number + 1,
        "trials": N_TRIALS,
        "training_timesteps": TRIAL_TIMESTEPS,
        "evaluation_episodes": EVAL_EPISODES,
        "train_sizes": TRAIN_SIZES,
        "eval_sizes": EVAL_SIZES,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print("OPTUNA COMPLETE")
    print("\nBest Parameters:")
    print(json.dumps(study.best_params, indent=4))


if __name__ == "__main__":
    main()