from pathlib import Path
import time

from .evaluate_ppo import make_env, save_result


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEEDS = [42, 123, 999]
TEST_SIZES = [5, 10, 15, 20, 30, 40]
EVAL_SEEDS = list(range(1000, 1250))


def evaluate_random_agent(test_size, eval_seeds, action_seed):
    env = make_env(test_size)
    successes = 0
    timeouts = 0
    other_outcomes = 0
    total_reward = 0
    total_steps = 0
    total_episodes = len(eval_seeds)
    episodes_run = 0
    start_time = time.time()

    for eval_seed in eval_seeds:
        obs, info = env.reset(seed=eval_seed)
        env.action_space.seed(eval_seed * 1000 + action_seed)
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        episodes_run += 1
        total_reward += episode_reward
        total_steps += steps
        success = episode_reward > 0
        if success:
            successes += 1
        elif truncated:
            timeouts += 1
        else:
            other_outcomes += 1

    if episodes_run != total_episodes:
        raise RuntimeError(
            f"Ran {episodes_run} episodes; expected {total_episodes}."
        )
    if successes + timeouts + other_outcomes != total_episodes:
        raise RuntimeError(
            "Episode outcome counts do not match the total: "
            f"{successes} successes + {timeouts} timeouts + "
            f"{other_outcomes} other != {total_episodes} episodes."
        )

    evaluation_time = time.time() - start_time
    env.close()

    return {
        "success_rate": successes / total_episodes,
        "average_reward": total_reward / total_episodes,
        "average_steps": total_steps / total_episodes,
        "timeout_rate": timeouts / total_episodes,
        "evaluation_time_seconds": evaluation_time,
    }


if __name__ == "__main__":
    for seed in SEEDS:
        model_name = f"random_action_seed{seed}"

        for test_size in TEST_SIZES:
            results = evaluate_random_agent(
                test_size=test_size,
                eval_seeds=EVAL_SEEDS,
                action_seed=seed,
            )

            save_result(
                algorithm="RANDOM_ACTION",
                model_name=model_name,
                train_size="N/A",
                test_size=test_size,
                timesteps=0,
                seed=seed,
                episodes=len(EVAL_SEEDS),
                results_dict=results,
                project_root=PROJECT_ROOT,
            )

            print(
                f"Random action seed {seed}"
                f" -> Test {test_size}x{test_size}"
                f" | Success = {results['success_rate']:.3f}"
                f" | Timeout = {results['timeout_rate']:.3f}"
            )

    print("\nALL RANDOM-AGENT EVALUATIONS COMPLETE")
