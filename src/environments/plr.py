from __future__ import annotations

import math
import random

import gymnasium as gym

from minigrid.wrappers import FlatObsWrapper

from .custom_empty import VariableSizeEmptyEnv


UNIFORM_SAMPLING_PROBABILITY = 0.20
PRIORITY_SUCCESS_MULTIPLIER = 0.95
PRIORITY_FAILURE_MULTIPLIER = 1.05
MIN_PRIORITY = 0.1
MAX_PRIORITY = 10.0


class PLREnv(gym.Env):
    """Mixed-size environment with simple adaptive size priorities."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        sizes=None,
        seed=None,
        uniform_sampling_probability=UNIFORM_SAMPLING_PROBABILITY,
    ):
        super().__init__()

        if sizes is None:
            sizes = [5, 10, 15, 20]

        self.sizes = list(sizes)
        if not self.sizes:
            raise ValueError("sizes must not be empty.")
        if not 0.0 <= uniform_sampling_probability <= 1.0:
            raise ValueError(
                "uniform_sampling_probability must be between 0 and 1."
            )

        self.uniform_sampling_probability = uniform_sampling_probability
        self.rng = random.Random(seed)
        self.episode_seed_rng = random.Random(seed)
        self.level_stats = {
            size: {
                "times_sampled": 0,
                "successes": 0,
                "failures": 0,
                "priority": 1.0,
            }
            for size in self.sizes
        }
        self.completed_episodes = 0
        self.episode_finished = False

        self.current_size = self.sizes[0]
        self.episode_reward = 0.0
        self.env = self._make_env(self.current_size)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    @staticmethod
    def _make_env(size):
        return FlatObsWrapper(VariableSizeEmptyEnv(size=size))

    def _select_size(self):
        if self.rng.random() < self.uniform_sampling_probability:
            return self.rng.choice(self.sizes)

        weights = [
            self.level_stats[size]["priority"]
            for size in self.sizes
        ]
        return self.rng.choices(
            self.sizes,
            weights=weights,
            k=1,
        )[0]

    def _validate_priorities(self):
        for size in self.sizes:
            priority = self.level_stats[size]["priority"]
            if (
                not math.isfinite(priority)
                or not MIN_PRIORITY <= priority <= MAX_PRIORITY
            ):
                raise RuntimeError(
                    f"Priority for size {size} must be finite and within "
                    f"[{MIN_PRIORITY}, {MAX_PRIORITY}]."
                )

    def _record_episode_result(self):
        stats = self.level_stats[self.current_size]
        self._validate_priorities()
        if self.episode_reward > 0:
            stats["successes"] += 1
            priority_multiplier = PRIORITY_SUCCESS_MULTIPLIER
        else:
            stats["failures"] += 1
            priority_multiplier = PRIORITY_FAILURE_MULTIPLIER

        stats["priority"] = min(
            MAX_PRIORITY,
            max(MIN_PRIORITY, stats["priority"] * priority_multiplier),
        )
        self._validate_priorities()
        self.completed_episodes += 1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_finished = False
        if seed is not None:
            self.rng.seed(seed)
            self.episode_seed_rng.seed(seed)
        self.env.close()

        self.current_size = self._select_size()
        self.episode_reward = 0.0
        self.env = self._make_env(self.current_size)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        episode_seed = self.episode_seed_rng.getrandbits(32)
        observation = self.env.reset(seed=episode_seed, options=options)

        self.level_stats[self.current_size]["times_sampled"] += 1
        return observation

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward

        if (terminated or truncated) and not self.episode_finished:
            self._record_episode_result()
            self.episode_finished = True

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def get_size_counts(self):
        return {
            size: stats["times_sampled"]
            for size, stats in self.level_stats.items()
        }

    def get_actual_distribution(self):
        counts = self.get_size_counts()
        total = sum(counts.values())
        if total == 0:
            return {str(size): 0.0 for size in self.sizes}

        return {
            str(size): counts[size] / total
            for size in self.sizes
        }

    def get_level_stats(self):
        return {
            size: dict(stats)
            for size, stats in self.level_stats.items()
        }
