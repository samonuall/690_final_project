import argparse
import logging
import os
import yaml

from src.logger import setup_logging
from src.envs.gridworld_wrapper import LLMRewardGridworld, REWARD_FNS, FITNESS_FNS
from src.training.train import train_ppo
from src.plot import plot_training

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to YAML config file (e.g. configs/config.yaml)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = os.path.join(cfg["output_dir"], "test")
    setup_logging(output_dir)
    log.info("Config loaded: %s", cfg)

    env = LLMRewardGridworld(
        env_name=cfg["env"]["env_name"],
        max_iterations=cfg["env"]["max_iterations"],
    )

    reward_fn = REWARD_FNS[cfg["env"]["reward_fn"]]
    fitness_fn = FITNESS_FNS[cfg["env"]["fitness_fn"]]

    model = train_ppo(env, reward_fn, fitness_fn, cfg, output_dir)

    model_path = os.path.join(output_dir, "ppo_policy")
    model.save(model_path)
    log.info("Model saved to %s.zip", model_path)

    plot_training(cfg["output_dir"])
    log.info("Done.")


if __name__ == "__main__":
    main()
