import numpy as np

def reward_fn(prev_obs, action, next_obs, info):
    """
    Reward shaping function for the SpillLab environment to assist PPO training.
    Encourages the agent to find the extraction point 'Y' (3.0) efficiently
    while penalizing steps and avoiding hazards.

    Args:
        prev_obs (np.ndarray): Board observation before action. Shape (1, 7, 9), float32.
        action (int): Action index (0: NOOP, 1: Up, 2: Down, 3: Left, 4: Right).
        next_obs (np.ndarray): Board observation after action. Shape (1, 7, 9), float32.
        info (dict): Environment info, must contain 'env_reward'.

    Returns:
        dict: {'total': float} reward used for training, plus break-down keys.
    """
    # Constants for reward shaping
    STEP_PENALTY = -0.1
    DIST_SHAPING = 1.0       # Coefficient for distance reduction reward
    TERM_SCALE = 5.0         # Scale sparse env_reward to ensure dominance
    
    # 1. Retrieve sparse environment reward (e.g., 1.0 for success, -1.0 for failure)
    # This signal comes from the base environment when the episode ends (reaches X or Y).
    env_reward = info.get("env_reward", 0.0)
    
    # Identify Static Target 'Y' (value 3.0) position
    # np.where returns tuple of arrays (depth_indices, row_indices, col_indices)
    # For shape (1, 7, 9), it returns arrays for dimension 0, 1, 2 respectively.
    target_indices = np.where(prev_obs == 3.0)
    
    # Check if target exists in observation (should always be present in valid maps)
    if len(target_indices[0]) == 0:
        return {"total": 0.0, "step_penalty": STEP_PENALTY}

    tr, tc = target_indices[1][0], target_indices[2][0]
    
    # Identify Agent 'A' (value 2.0) position in previous step
    agent_prev_indices = np.where(prev_obs == 2.0)
    
    # Identify Agent 'A' (value 2.0) position in next step
    agent_next_indices = np.where(next_obs == 2.0)
    
    total_reward = 0.0
    
    # Case A: Episode has ended (Terminal Step)
    # If env_reward is non-zero, prioritize the environment's sparse signal.
    # We scale it up to create significant gradients without overwhelming the step cost.
    if env_reward != 0.0:
        terminal_score = env_reward * TERM_SCALE
        # Optionally include step penalty for the final action
        total_reward = terminal_score + STEP_PENALTY
        return {
            "total": total_reward,
            "env_reward_score": env_reward,
            "step_penalty": STEP_PENALTY
        }
    
    # Case B: Intermediate Step (Episode not terminal)
    # Apply dense shaping rewards to encourage learning.
    shaping_reward = 0.0
    
    # Ensure agent positions are found (valid assumption for running episode)
    if len(agent_prev_indices[0]) > 0 and len(agent_next_indices[0]) > 0:
        # Extract coordinates
        # agent_prev_indices structure: (depth_arr, row_arr, col_arr)
        ar_prev, ac_prev = agent_prev_indices[1][0], agent_prev_indices[2][0]
        ar_next, ac_next = agent_next_indices[1][0], agent_next_indices[2][0]
        
        # Calculate Manhattan distances to target
        dist_prev = abs(ar_prev - tr) + abs(ac_prev - tc)
        dist_next = abs(ar_next - tr) + abs(ac_next - tc)
        
        # Distance Shaping: Reward for reducing distance to target
        # (prev_dist - next_dist) is positive if closer, negative if further.
        shaping_reward = DIST_SHAPING * (dist_prev - dist_next)
        
    # Step Penalty is applied to every action to encourage efficiency
    total_reward = STEP_PENALTY + shaping_reward
    
    return {
        "total": total_reward,
        "step_penalty": STEP_PENALTY,
        "distance_reward": shaping_reward
    }