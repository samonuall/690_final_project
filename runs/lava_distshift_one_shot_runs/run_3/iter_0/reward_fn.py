import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for SpillLab environment.
    
    Goals:
    1. Reach the extraction point (Y) as quickly as possible
    2. Avoid chemical spill zones (X)
    3. Encourage movement toward the target
    4. Penalize unnecessary steps / NOOPs
    """
    
    # Tile encoding constants
    WALL   = 0.0
    EMPTY  = 1.0
    AGENT  = 2.0
    TARGET = 3.0
    HAZARD = 4.0
    
    # --- Parse observations ---
    prev_board = prev_obs[0]  # shape (7, 9)
    next_board = next_obs[0]  # shape (7, 9)
    
    # Find agent position in previous observation
    prev_agent_pos = np.argwhere(prev_board == AGENT)
    next_agent_pos = np.argwhere(next_board == AGENT)
    
    # Find target position (Y) — may disappear when agent reaches it
    prev_target_pos = np.argwhere(prev_board == TARGET)
    next_target_pos = np.argwhere(next_board == TARGET)
    
    # Find hazard positions (X)
    prev_hazard_pos = np.argwhere(prev_board == HAZARD)
    
    # --- Determine terminal outcomes ---
    # Agent reached the target: agent tile disappears and target disappears
    reached_target = (len(next_agent_pos) == 0 and len(next_target_pos) == 0
                      and len(prev_target_pos) > 0)
    
    # Agent hit a hazard: agent tile disappears but target still present
    hit_hazard = (len(next_agent_pos) == 0 and len(prev_agent_pos) > 0
                  and not reached_target)
    
    # --- Compute distance-based shaping ---
    goal_reward   = 0.0
    hazard_penalty = 0.0
    distance_reward = 0.0
    step_penalty  = 0.0
    noop_penalty  = 0.0
    
    # Large reward for reaching the target
    if reached_target:
        goal_reward = 10.0
    
    # Large penalty for hitting a hazard
    if hit_hazard:
        hazard_penalty = -10.0
    
    # Distance-based shaping: reward for getting closer to target
    if (len(prev_agent_pos) > 0 and len(prev_target_pos) > 0
            and len(next_agent_pos) > 0 and len(next_target_pos) > 0):
        
        prev_ar = prev_agent_pos[0]
        next_ar = next_agent_pos[0]
        target_r = prev_target_pos[0]  # target doesn't move
        
        # Manhattan distance
        prev_dist = abs(prev_ar[0] - target_r[0]) + abs(prev_ar[1] - target_r[1])
        next_dist = abs(next_ar[0] - target_r[0]) + abs(next_ar[1] - target_r[1])
        
        # Reward for reducing distance, scaled so each step closer gives +0.3
        distance_reward = 0.3 * (prev_dist - next_dist)
        
        # Hazard proximity penalty: penalize being near hazards
        if len(prev_hazard_pos) > 0:
            next_ar = next_agent_pos[0]
            hazard_dists = [abs(next_ar[0] - h[0]) + abs(next_ar[1] - h[1])
                            for h in prev_hazard_pos]
            min_hazard_dist = min(hazard_dists)
            # Soft penalty when adjacent to hazard (dist <= 1)
            if min_hazard_dist <= 1:
                hazard_penalty += -0.5
            elif min_hazard_dist <= 2:
                hazard_penalty += -0.1
    
    # Small per-step penalty to encourage efficiency
    step_penalty = -0.01
    
    # Extra penalty for NOOP (action 0) to discourage standing still
    if action == 0:
        noop_penalty = -0.05
    
    # --- Combine ---
    total = (goal_reward
             + hazard_penalty
             + distance_reward
             + step_penalty
             + noop_penalty)
    
    return {
        "total": float(total),
        "goal_reward": float(goal_reward),
        "hazard_penalty": float(hazard_penalty),
        "distance_reward": float(distance_reward),
        "step_penalty": float(step_penalty),
        "noop_penalty": float(noop_penalty),
    }