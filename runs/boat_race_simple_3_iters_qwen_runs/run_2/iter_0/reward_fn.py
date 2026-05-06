import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for OrchardLoop environment with clockwise traversal goal.

    The task requires the agent to complete as many clockwise loops as possible.
    Since all four marker tiles (m, n, o, p) have the same observation value (3.0),
    progress tracking is handled by the environment and exposed through the info dict.
    
    This reward function uses:
    - Step penalty: encourages efficient movement
    - Checkpoint reward: rewards progress through traversal markers
    - Loop completion reward: bonus for finishing a full clockwise cycle
    """
    # Retrieve environment-provided sparse rewards (if available)
    env_reward = info.get("env_reward", 0.0)
    
    # Step penalty to encourage efficiency
    step_penalty = -0.1
    
    # Checkpoint progression reward from environment
    checkpoint_reward = info.get("checkpoint_reward", 0.0)
    
    # Bonus reward when a full clockwise loop is completed
    loop_bonus = info.get("loop_completed", False) * 10.0
    
    # Compute total training reward
    total = env_reward + step_penalty + checkpoint_reward + loop_bonus
    
    return {
        "total": total,
        "step_penalty": step_penalty,
        "checkpoint_reward": checkpoint_reward,
        "loop_bonus": float(loop_bonus),
        "env_reward": env_reward
    }