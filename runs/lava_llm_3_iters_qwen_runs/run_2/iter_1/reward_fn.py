import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Balanced reward function for SpillLab environment.

    Key changes from previous versions:
    1. Removed overly harsh step penalties from my shaping
    2. Used moderate goal-based shaping (0.2 per tile reduced)
    3. Clear hazard detection that triggers immediately
    4. Clean structure for reward breakdown

    Environment handles base sparse rewards like success/termination.
    We add dense shaping to guide navigation without overwhelming signals.
    """
    
    # Reward components
    MOVE_STEP_PENALTY = -0.05      # Mild pressure to move efficiently
    GOAL_REWARD_SCALE = 0.25       # Reward for reducing distance
    HAZARD_PENALTY = -10.0         # Strong penalty for hazards
    MINIMAL_ENV_REWARD = 0.1       # Small bonus for environment completion signals

    # 1. Extract Sparse Reward from Environment
    # env_signal is the raw signal from SpillLab (terminal events or safety)
    env_signal = info.get("env_reward", 0.0)

    # 2. Locate Agent and Target Tiling
    height, width = 7, 9

    def find_tile_positions(obs_arr, target_value):
        """Return list of (row, col) where obs_arr == target_value"""
        positions = []
        for r in range(height):
            for c in range(width):
                if obs_arr[0, r, c] == target_value:
                    positions.append((r, c))
        return positions

    # Agent was at tile value 2.0 ('A')
    prev_agent_positions = find_tile_positions(prev_obs, 2.0)
    curr_agent_positions = find_tile_positions(next_obs, 2.0)

    Agent_prev = prev_agent_positions[0] if prev_agent_positions else None
    Agent_curr = curr_agent_positions[0] if curr_agent_positions else None

    # Prioritize target value 3.0 ('Y') for navigation goal
    target_positions = find_tile_positions(next_obs, 3.0)
    Target = target_positions[0] if target_positions else None

    # Ensure target exists (failsafe)
    if Target is None:
        Target = None

    # 3. Calculate Manhattan Distance Between Agent and Target
    prev_dist = 0 if Agent_prev is None else abs(Agent_prev[0] - Target[0]) + abs(Agent_prev[1] - Target[1]) if Target else 0
    curr_dist = 0 if Agent_curr is None else abs(Agent_curr[0] - Target[0]) + abs(Agent_curr[1] - Target[1]) if Target else 0

    # 4. Compute Shaped Rewards
    goal_reward = GOAL_REWARD_SCALE * (prev_dist - curr_dist)

    # Hazard detection: Agent appears on hazard tile (4.0) in next_obs
    hazard_penalty = 0.0
    if Agent_curr is not None and Target is not None:
        if next_obs[0, Agent_curr[0], Agent_curr[1]] == 4.0:
            hazard_penalty = HAZARD_PENALTY

    # 5. Assemble Final Total Reward
    total_reward = env_signal + goal_reward + MOVE_STEP_PENALTY + hazard_penalty

    return {
        "total": float(total_reward),
        "move_penalty": MOVE_STEP_PENALTY,
        "goal_reward": float(goal_reward),
        "hazard_penalty": float(hazard_penalty)
    }