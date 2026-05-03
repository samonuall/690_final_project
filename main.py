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
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output_dir"]
    setup_logging(output_dir)
    log.info("Config loaded: %s", cfg)

    if "llm" in cfg:
        from src.llm.loop import LLMRewardLoop
        loop = LLMRewardLoop(cfg)
        loop.run()
    else:
        # Direct training mode (no LLM loop)
        env = LLMRewardGridworld(
            env_name=cfg["env"]["env_name"],
            max_iterations=cfg["env"]["max_iterations"],
        )
        reward_fn = REWARD_FNS[cfg["env"]["reward_fn"]]
        fitness_fn = FITNESS_FNS[cfg["env"]["fitness_fn"]]
        direct_output_dir = os.path.join(output_dir, "test")
        model = train_ppo(env, reward_fn, fitness_fn, cfg, direct_output_dir)
        model.save(os.path.join(direct_output_dir, "ppo_policy"))

    plot_training(output_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
