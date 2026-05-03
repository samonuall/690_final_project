import importlib.util
import json
import logging
import traceback
from pathlib import Path

from src.envs.gridworld_wrapper import LLMRewardGridworld, FITNESS_FNS
from src.training.train import train_ppo
from .client import OpenRouterClient
from .prompts import build_system_prompt, build_feedback_message, extract_python_code

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3


class LLMRewardLoop:
    """
    Orchestrates the LLM → reward function → PPO training → feedback loop.

    For each iteration:
      1. Send context (and prior training feedback) to the LLM.
      2. Extract a Python reward function from the response.
      3. Smoke-test the function, reprompting on error (up to _MAX_RETRIES total attempts).
      4. Run full PPO training with the accepted function.
      5. Build structured feedback from the training log for the next round.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.llm_cfg = cfg["llm"]
        self.output_dir = Path(cfg["output_dir"])
        self.client = OpenRouterClient(self.llm_cfg["model"])
        self.n_iterations: int = self.llm_cfg["n_iterations"]

    def run(self) -> None:
        env_context = Path(self.llm_cfg["context_file"]).read_text()
        obfuscated_code_path = self.llm_cfg.get("obfuscated_code_file")
        obfuscated_code = Path(obfuscated_code_path).read_text() if obfuscated_code_path else ""
        system_prompt = build_system_prompt(
            self.llm_cfg["system_prompt_file"], env_context, obfuscated_code
        )
        self.client.add_system(system_prompt)

        fitness_fn = FITNESS_FNS[self.cfg["env"]["fitness_fn"]]
        training_log: dict | None = None
        iteration_records: list[dict] = []

        for i in range(self.n_iterations):
            logger.info("=== LLM iteration %d / %d ===", i + 1, self.n_iterations)
            iter_dir = self.output_dir / f"iter_{i}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            user_msg = (
                "Design a reward function for this environment."
                if i == 0
                else build_feedback_message(training_log)
            )

            send_idx_before = len(self.client.token_usage)
            reward_fn, code, n_attempts = self._get_reward_fn(user_msg, iter_dir, i)
            send_idx_after = len(self.client.token_usage)

            logger.info("Running PPO training for iteration %d", i)
            env = self._make_env()
            model = train_ppo(env, reward_fn, fitness_fn, self.cfg, str(iter_dir))
            model.save(str(iter_dir / "ppo_policy"))

            with open(iter_dir / "training_log.json") as f:
                training_log = json.load(f)

            round_tokens = self.client.token_usage[send_idx_before:send_idx_after]
            iteration_records.append({
                "iteration": i,
                "reward_fn_code": code,
                "n_attempts": n_attempts,
                "round_token_usage": round_tokens,
                "cumulative_tokens_after_round": self.client.cumulative_tokens,
            })
            logger.info("Iteration %d complete. Cumulative tokens: %s", i, self.client.cumulative_tokens)

        llm_log = {
            "iterations": iteration_records,
            "full_conversation": self.client.messages,
            "total_tokens": self.client.cumulative_tokens,
        }
        log_path = self.output_dir / "llm_log.json"
        with open(log_path, "w") as f:
            json.dump(llm_log, f, indent=2)
        logger.info("LLM log saved to %s", log_path)

    def _get_reward_fn(
        self, initial_message: str, iter_dir: Path, iteration: int
    ) -> tuple:
        message = initial_message
        for attempt in range(_MAX_RETRIES):
            response = self.client.send(message)
            code = extract_python_code(response)

            if code is None:
                message = (
                    "Error: no Python code block found in your response. "
                    "Please provide your reward function inside a ```python ... ``` fenced block."
                )
                logger.warning("iter %d attempt %d: no code block in response", iteration, attempt)
                continue

            reward_fn_path = iter_dir / "reward_fn.py"
            reward_fn_path.write_text(code)

            module_name = f"llm_reward_iter{iteration}_attempt{attempt}"
            try:
                reward_fn = _load_fn(reward_fn_path, module_name)
            except Exception:
                tb = traceback.format_exc()
                message = (
                    f"Error: failed to import your reward function:\n\n```\n{tb}\n```\n\n"
                    "Please fix the error and provide the corrected function."
                )
                logger.warning("iter %d attempt %d: import error\n%s", iteration, attempt, tb)
                continue

            # Smoke-test with a real env step so we catch shape/type issues early.
            try:
                test_env = self._make_env()
                obs, _ = test_env.reset()
                action = int(test_env.action_space.sample())
                next_obs, _, _, _, info = test_env.step(action)
                result = reward_fn(obs, action, next_obs, info)
                if not isinstance(result, dict) or "total" not in result:
                    raise TypeError(
                        f"reward_fn must return a dict with a 'total' key, got: {result!r}"
                    )
            except Exception:
                tb = traceback.format_exc()
                message = (
                    f"Error: your reward function failed a validation test:\n\n```\n{tb}\n```\n\n"
                    "Please fix the error and provide the corrected function."
                )
                logger.warning("iter %d attempt %d: smoke-test failed\n%s", iteration, attempt, tb)
                continue

            logger.info("iter %d: accepted reward_fn after %d attempt(s)", iteration, attempt + 1)
            return reward_fn, code, attempt + 1

        raise RuntimeError(
            f"Failed to get a valid reward function after {_MAX_RETRIES} attempts "
            f"in iteration {iteration}. Check llm_log.json for the full conversation."
        )

    def _make_env(self) -> LLMRewardGridworld:
        env_cfg = self.cfg["env"]
        return LLMRewardGridworld(
            env_name=env_cfg["env_name"],
            max_iterations=env_cfg.get("max_iterations", 100),
        )


def _load_fn(filepath: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "reward_fn"):
        raise AttributeError(f"No top-level 'reward_fn' function found in {filepath}")
    return module.reward_fn
