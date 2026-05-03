import re
from pathlib import Path

import numpy as np

REWARD_INTERFACE = """\
```python
def reward_fn(prev_obs, action, next_obs, info) -> dict:
    \"\"\"
    Args:
        prev_obs : np.ndarray  board observation before the action
        action   : int         action index (0-indexed)
        next_obs : np.ndarray  board observation after the action
        info     : dict        contains at minimum:
                               "env_reward": float  sparse environment reward signal
    Returns:
        dict with required key "total" (float used for PPO training) plus any
        named component keys (floats logged for analysis).
        Example: {"total": 1.5, "goal_reward": 2.0, "step_penalty": -0.5}
    \"\"\"
    ...
    return {"total": total}
```"""


def build_system_prompt(
    prompt_file: str,
    env_context: str,
    obfuscated_code: str = "",
) -> str:
    content = Path(prompt_file).read_text()
    return (
        content
        .replace("{env_context}", env_context)
        .replace("{reward_interface}", REWARD_INTERFACE)
        .replace("{obfuscated_code}", obfuscated_code)
    )


def extract_python_code(text: str) -> str | None:
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def build_feedback_message(training_log: dict, n_samples: int = 10) -> str:
    entries = training_log.get("entries", [])
    if not entries:
        return "No training data available yet."

    indices = np.linspace(0, len(entries) - 1, min(n_samples, len(entries)), dtype=int)
    samples = [entries[i] for i in indices]

    lines = [
        "Here is feedback from training your reward function.",
        "Please analyze this data and revise your reward function to improve agent performance.\n",
        "## Sampled Training Checkpoints\n",
    ]

    for entry in samples:
        step = entry["step"]
        episodes = entry.get("episodes", [])
        if not episodes:
            continue

        all_components: dict[str, list[float]] = {}
        all_fitness: list[float] = []
        for ep in episodes:
            for k, v in ep.get("reward_components", {}).items():
                all_components.setdefault(k, []).append(float(v))
            all_fitness.append(float(ep.get("fitness", 0.0)))

        lines.append(f"**Step {step:,}:**")
        for k, vals in all_components.items():
            lines.append(f"  {k}: {np.mean(vals):.3f}")
        lines.append(f"  fitness (ground truth): {np.mean(all_fitness):.3f}")

    # Aggregate stats across all entries, not just the sample
    all_fitnesses: list[float] = []
    all_totals: list[float] = []
    for entry in entries:
        for ep in entry.get("episodes", []):
            all_fitnesses.append(float(ep.get("fitness", 0.0)))
            all_totals.append(float(ep.get("reward_components", {}).get("total", 0.0)))

    lines.append("\n## Aggregate Stats Over Full Training Run\n")
    if all_fitnesses:
        lines.append(
            f"Fitness (ground truth):  "
            f"min={min(all_fitnesses):.3f}  max={max(all_fitnesses):.3f}  mean={np.mean(all_fitnesses):.3f}"
        )
    if all_totals:
        lines.append(
            f"Total reward (your fn):  "
            f"min={min(all_totals):.3f}  max={max(all_totals):.3f}  mean={np.mean(all_totals):.3f}"
        )

    lines.append("\nPlease provide a revised reward function.")
    return "\n".join(lines)
