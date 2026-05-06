import json
import logging
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class _EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, eval_episodes, log, total_timesteps):
        super().__init__(verbose=0)
        self._eval_env = eval_env
        self._eval_freq = eval_freq
        self._eval_episodes = eval_episodes
        self._log = log  # shared list, mutated in-place
        self._total_timesteps = total_timesteps
        self._log_freq = max(1, total_timesteps // 10)  # progress log every ~10%
        # Running totals accumulated during training steps (not eval rollouts)
        self._cum_training_reward = 0.0
        self._cum_training_fitness = 0.0

    def _on_step(self):
        if self.n_calls % self._log_freq == 0:
            pct = 100 * self.num_timesteps / self._total_timesteps
            logger.info("Training progress: %d/%d steps (%.0f%%)",
                        self.num_timesteps, self._total_timesteps, pct)

        # Accumulate per-step reward and fitness from the live training batch
        rewards = self.locals.get("rewards", [])
        infos = self.locals.get("infos", [{}])
        self._cum_training_reward += float(np.sum(rewards)) if len(rewards) else 0.0
        self._cum_training_fitness += sum(
            float(info.get("fitness", 0.0)) for info in infos
        )

        if self.n_calls % self._eval_freq != 0:
            return True

        self.model.policy.set_training_mode(False)
        episodes = []

        for _ in range(self._eval_episodes):
            obs, _ = self._eval_env.reset()
            done = False
            ep_components = {}
            ep_fitness = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self._eval_env.step(int(action))
                done = terminated or truncated

                for k, v in info.get("reward_components", {}).items():
                    ep_components[k] = ep_components.get(k, 0.0) + float(v)
                ep_fitness += info.get("fitness", 0.0)

            episodes.append({"reward_components": ep_components, "fitness": ep_fitness})

        # Snapshot the latest PPO loss stats (populated after each train() call)
        ntv = self.model.logger.name_to_value
        ppo_stats = {
            k.replace("train/", ""): float(v)
            for k in [
                "train/policy_gradient_loss",
                "train/value_loss",
                "train/entropy_loss",
                "train/explained_variance",
                "train/approx_kl",
                "train/clip_fraction",
            ]
            if (v := ntv.get(k)) is not None
        }

        self._log.append({
            "step": self.num_timesteps,
            "episodes": episodes,
            "cumulative_training_reward": self._cum_training_reward,
            "cumulative_training_fitness": self._cum_training_fitness,
            "ppo_stats": ppo_stats,
        })
        self.model.policy.set_training_mode(True)

        mean_fitness = np.mean([ep["fitness"] for ep in episodes])
        mean_reward = np.mean([ep["reward_components"].get("total", 0.0) for ep in episodes])

        logger.info(
            "eval step=%d  mean_fitness=%.3f  mean_reward_total=%.3f  "
            "value_loss=%.4f  explained_var=%.4f  entropy=%.4f",
            self.num_timesteps, mean_fitness, mean_reward,
            ppo_stats.get("value_loss", float("nan")),
            ppo_stats.get("explained_variance", float("nan")),
            ppo_stats.get("entropy_loss", float("nan")),
        )
        return True


def train_ppo(env, reward_fn, fitness_fn, cfg, output_dir):
    """
    Train a PPO agent using reward_fn as the training signal.
    Logs reward components and fitness at eval checkpoints to output_dir/training_log.json.

    Args:
        env: LLMRewardGridworld instance
        reward_fn: callable(obs, action, next_obs, info) -> dict with 'total' key
        fitness_fn: callable(obs, action, next_obs, info) -> float
        cfg: config dict loaded from config.yaml
        output_dir: directory to save training_log.json

    Returns:
        Trained PPO model
    """
    env.set_reward_fn(reward_fn)
    env.set_fitness_fn(fitness_fn)

    ppo_cfg = cfg["ppo"]
    seed = cfg["seed"]

    logger.info(
        "Starting PPO training: total_timesteps=%d  eval_freq=%d  eval_episodes=%d  "
        "net_arch=%s  lr=%s  gamma=%s  seed=%d  output_dir=%s",
        ppo_cfg["total_timesteps"], ppo_cfg["eval_freq"], ppo_cfg["eval_episodes"],
        ppo_cfg["net_arch"], ppo_cfg["learning_rate"], ppo_cfg.get("gamma", 0.99),
        seed, output_dir,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        gamma=ppo_cfg.get("gamma", 0.99),
        ent_coef=ppo_cfg.get("ent_coef", 0.0),
        policy_kwargs=dict(net_arch=ppo_cfg["net_arch"]),
        seed=seed,
        verbose=0,
    )

    os.makedirs(output_dir, exist_ok=True)
    log = []
    callback = _EvalCallback(
        eval_env=env,
        eval_freq=ppo_cfg["eval_freq"],
        eval_episodes=ppo_cfg["eval_episodes"],
        log=log,
        total_timesteps=ppo_cfg["total_timesteps"],
    )

    model.learn(total_timesteps=ppo_cfg["total_timesteps"], callback=callback)

    result = {
        "total_timesteps": ppo_cfg["total_timesteps"],
        "eval_freq": ppo_cfg["eval_freq"],
        "entries": log,
    }
    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Training complete. Log saved to %s", log_path)

    return model
