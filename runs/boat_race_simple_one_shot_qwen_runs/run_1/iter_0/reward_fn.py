import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for the OrchardLoop environment.

    The environment provides sparse rewards for passing markers in the loop
    through info['env_reward']. We amplify this signal slightly and add
    a small step penalty to encourage efficient completion of loops.

    Args:
        prev_obs : np.ndarray  board observation before the action
        action   : int         action index (0-indexed)
        next_obs : np.ndarray  board observation after the action
        info     : dict        sparse environment reward signal

    Returns:
        dict with key "total" used for PPO training, plus "step_reward" and
        "mark_reward" for logging.
    """
    env_reward = float(info.get("env_reward", 0.0))
    step_penalty = -0.05

    # Combine environment's progress reward with small step penalty
    total = env_reward + step_penalty

    return {
        "total": total,
        "step_reward": step_penalty,
        "mark_reward": env_reward,
    }