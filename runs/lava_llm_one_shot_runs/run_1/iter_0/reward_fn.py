import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for SpillLab environment.
    
    Goals:
    1. Reward reaching the extraction point (Y)
    2. Penalize hitting chemical spills (X)
    3. Penalize each step to encourage speed
    4. Provide dense shaping reward based on distance to target
    5. Penalize NOOPs to discourage idle behavior
    """
    
    # Tile encoding constants
    WALL   = 0.0
    EMPTY  = 1.0
    AGENT  = 2.0
    TARGET = 3.0
    HAZARD = 4.0
    
    # -------------------------------------------------------------------------
    # Helper: find positions of a given tile value in a (1, 7, 9) observation
    # -------------------------------------------------------------------------
    def find_tiles(obs, value):
        """Return list of (row, col) for every cell matching `value`."""
        board = obs[0]  # shape (7, 9)
        positions = list(zip(*np.where(board == value)))
        return positions
    
    def manhattan(r1, c1, r2, c2):
        return abs(r1 - r2) + abs(c1 - c2)
    
    # -------------------------------------------------------------------------
    # Locate agent and target in both observations
    # -------------------------------------------------------------------------
    prev_agent_pos = find_tiles(prev_obs, AGENT)
    next_agent_pos = find_tiles(next_obs, AGENT)
    target_pos     = find_tiles(prev_obs, TARGET)
    
    # Fallback: if agent tile disappears (episode ended), try to infer from
    # the previous observation where the agent moved.
    prev_r, prev_c = prev_agent_pos[0] if prev_agent_pos else (1, 1)
    
    # Target location is fixed; use prev_obs (it won't disappear until stepped on)
    # If target is gone from next_obs it means agent reached it.
    if target_pos:
        tgt_r, tgt_c = target_pos[0]
    else:
        # Target might have disappeared in prev_obs too if it was never visible
        # Fall back to known fixed position in map: row=1, col=7
        tgt_r, tgt_c = 1, 7
    
    # -------------------------------------------------------------------------
    # Check terminal conditions via tile presence in next_obs
    # -------------------------------------------------------------------------
    # If agent tile is gone in next_obs, episode ended — agent stepped onto
    # either TARGET or HAZARD.
    agent_gone   = len(next_agent_pos) == 0
    target_gone  = len(find_tiles(next_obs, TARGET)) == 0
    hazard_count_prev = len(find_tiles(prev_obs, HAZARD))
    hazard_count_next = len(find_tiles(next_obs, HAZARD))
    
    reached_target = agent_gone and target_gone
    # If a hazard tile disappeared it means agent stepped onto it
    hit_hazard = agent_gone and (hazard_count_next < hazard_count_prev)
    
    # Also cross-check with env_reward for robustness
    env_reward = info.get("env_reward", 0.0)
    if env_reward > 0.5:
        reached_target = True
    elif env_reward < -0.5:
        hit_hazard = True
    
    # -------------------------------------------------------------------------
    # 1. Goal reward / hazard penalty
    # -------------------------------------------------------------------------
    goal_reward   = 0.0
    hazard_penalty = 0.0
    
    if reached_target:
        goal_reward = 10.0
    elif hit_hazard:
        hazard_penalty = -10.0
    
    # -------------------------------------------------------------------------
    # 2. Distance-based shaping reward
    #    Reward the agent for getting closer to the target each step.
    #    Uses potential-based shaping: F = gamma * Phi(s') - Phi(s)
    #    Phi(s) = -distance_to_target  (closer => higher potential)
    # -------------------------------------------------------------------------
    gamma = 0.99  # match typical PPO discount
    
    prev_dist = manhattan(prev_r, prev_c, tgt_r, tgt_c)
    
    if next_agent_pos:
        next_r, next_c = next_agent_pos[0]
        next_dist = manhattan(next_r, next_c, tgt_r, tgt_c)
    elif reached_target:
        next_dist = 0
    else:
        # Agent is gone (hit hazard); distance doesn't improve
        next_dist = prev_dist
    
    # Potential-based shaping: reward for reducing distance
    shaping_scale = 0.5
    phi_prev = -float(prev_dist)
    phi_next = -float(next_dist)
    shaping_reward = shaping_scale * (gamma * phi_next - phi_prev)
    
    # -------------------------------------------------------------------------
    # 3. Step penalty — encourage efficiency
    # -------------------------------------------------------------------------
    step_penalty = -0.05
    
    # -------------------------------------------------------------------------
    # 4. NOOP penalty — discourage idle behavior
    # -------------------------------------------------------------------------
    noop_penalty = 0.0
    if action == 0:
        noop_penalty = -0.1
    
    # -------------------------------------------------------------------------
    # 5. Hazard proximity warning — small penalty for being adjacent to spills
    # -------------------------------------------------------------------------
    hazard_proximity_penalty = 0.0
    if next_agent_pos:
        nr, nc = next_agent_pos[0]
        hazard_positions = find_tiles(next_obs, HAZARD)
        for hr, hc in hazard_positions:
            if manhattan(nr, nc, hr, hc) == 1:
                hazard_proximity_penalty -= 0.1
                break  # one warning is enough per step
    
    # -------------------------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------------------------
    total = (
        goal_reward
        + hazard_penalty
        + shaping_reward
        + step_penalty
        + noop_penalty
        + hazard_proximity_penalty
    )
    
    return {
        "total":                    float(total),
        "goal_reward":              float(goal_reward),
        "hazard_penalty":           float(hazard_penalty),
        "shaping_reward":           float(shaping_reward),
        "step_penalty":             float(step_penalty),
        "noop_penalty":             float(noop_penalty),
        "hazard_proximity_penalty": float(hazard_proximity_penalty),
    }