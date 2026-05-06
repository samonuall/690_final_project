import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for SpillLab environment.
    
    Goals:
    1. Reward reaching the extraction point (Y)
    2. Penalize stepping into chemical spills (X)
    3. Penalize each step to encourage efficiency
    4. Provide dense shaping reward based on distance to target
    5. Penalize NOOP to discourage standing still
    """
    
    # Tile encoding constants
    WALL   = 0.0
    EMPTY  = 1.0
    AGENT  = 2.0
    TARGET = 3.0
    HAZARD = 4.0
    
    # --- Locate agent and target positions ---
    def find_positions(obs, value):
        """Return list of (row, col) for all cells matching value."""
        positions = np.argwhere(obs[0] == value)
        return positions
    
    prev_board = prev_obs
    next_board = next_obs
    
    # Find agent position in previous and next observations
    prev_agent_positions = find_positions(prev_board, AGENT)
    next_agent_positions = find_positions(next_board, AGENT)
    
    # Find target position (Y) — may disappear if agent reaches it
    prev_target_positions = find_positions(prev_board, TARGET)
    next_target_positions = find_positions(next_board, TARGET)
    
    # Find hazard positions
    prev_hazard_positions = find_positions(prev_board, HAZARD)
    
    # --- Determine terminal events ---
    env_reward = info.get("env_reward", 0.0)
    
    # Check if agent reached target (Y tile disappears or agent tile disappears)
    reached_target = False
    hit_hazard = False
    
    if len(prev_target_positions) > 0 and len(next_target_positions) == 0:
        # Target tile vanished → agent reached it
        reached_target = True
    
    if len(prev_agent_positions) > 0 and len(next_agent_positions) == 0:
        # Agent tile vanished — could be target or hazard
        if not reached_target:
            # Agent disappeared but target still exists → hit hazard
            hit_hazard = True
    
    # Also use env_reward as a signal
    if env_reward > 0.5:
        reached_target = True
    elif env_reward < -0.5:
        hit_hazard = True
    
    # --- Compute Manhattan distance to target ---
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    # Get agent positions (use board center if not found)
    if len(prev_agent_positions) > 0:
        prev_agent = prev_agent_positions[0]
    else:
        prev_agent = np.array([3, 4])  # fallback to center
    
    if len(next_agent_positions) > 0:
        next_agent = next_agent_positions[0]
    else:
        # If agent disappeared, estimate from previous position + action
        next_agent = prev_agent.copy()
    
    # Get target position
    if len(prev_target_positions) > 0:
        target_pos = prev_target_positions[0]
    elif len(next_target_positions) > 0:
        target_pos = next_target_positions[0]
    else:
        target_pos = prev_agent  # fallback
    
    # --- Distance-based shaping reward ---
    prev_dist = manhattan_distance(prev_agent, target_pos)
    next_dist = manhattan_distance(next_agent, target_pos)
    
    # Reward for getting closer to the target
    distance_improvement = prev_dist - next_dist
    shaping_reward = distance_improvement * 0.15
    
    # --- Hazard proximity penalty ---
    # Penalize being close to hazards to encourage safe navigation
    hazard_penalty = 0.0
    if len(prev_hazard_positions) > 0:
        # Penalize based on proximity to nearest hazard in next state
        min_hazard_dist_next = min(
            manhattan_distance(next_agent, h) for h in prev_hazard_positions
        )
        # Only penalize if very close (within 1-2 tiles)
        if min_hazard_dist_next <= 1:
            hazard_penalty = -0.2
        elif min_hazard_dist_next <= 2:
            hazard_penalty = -0.05
    
    # --- Terminal rewards ---
    goal_reward = 0.0
    hazard_hit_penalty = 0.0
    
    if reached_target:
        goal_reward = 5.0
    elif hit_hazard:
        hazard_hit_penalty = -5.0
    
    # --- Step penalty (encourages efficiency) ---
    step_penalty = -0.02
    
    # --- NOOP penalty ---
    noop_penalty = 0.0
    if action == 0:  # NOOP
        noop_penalty = -0.05
    
    # --- Assemble total reward ---
    total = (
        goal_reward
        + hazard_hit_penalty
        + shaping_reward
        + hazard_penalty
        + step_penalty
        + noop_penalty
    )
    
    return {
        "total": float(total),
        "goal_reward": float(goal_reward),
        "hazard_hit_penalty": float(hazard_hit_penalty),
        "shaping_reward": float(shaping_reward),
        "hazard_penalty": float(hazard_penalty),
        "step_penalty": float(step_penalty),
        "noop_penalty": float(noop_penalty),
    }