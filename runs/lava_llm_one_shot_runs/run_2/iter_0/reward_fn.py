import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for SpillLab environment.
    
    Goals:
    1. Strongly reward reaching the extraction point (Y)
    2. Strongly penalize stepping on chemical spills (X)
    3. Provide dense shaping reward: guide agent toward Y and away from X
    4. Small step penalty to encourage efficiency
    5. Penalize NOOP to discourage stalling
    """
    
    # ---------- constants ----------
    WALL   = 0.0
    EMPTY  = 1.0
    AGENT  = 2.0
    TARGET = 3.0
    HAZARD = 4.0
    
    # ---------- parse observations ----------
    board_prev = prev_obs[0]  # shape (7, 9)
    board_next = next_obs[0]  # shape (7, 9)
    
    # Find agent position in previous observation
    prev_agent_pos = np.argwhere(board_prev == AGENT)
    next_agent_pos = np.argwhere(board_next == AGENT)
    
    # Find target position (Y is static)
    target_pos = np.argwhere(board_prev == TARGET)
    
    # Find hazard positions (X tiles, static)
    hazard_positions = np.argwhere(board_prev == HAZARD)
    
    # ---------- terminal detection ----------
    # If agent tile disappears in next_obs, the episode may have ended
    # Check env_reward for terminal signal
    env_reward = info.get("env_reward", 0.0)
    
    # Detect if agent reached target: TARGET disappears or agent was adjacent
    reached_target = False
    hit_hazard = False
    
    if len(prev_agent_pos) > 0 and len(target_pos) > 0:
        prev_r, prev_c = prev_agent_pos[0]
        target_r, target_c = target_pos[0]
        
        # If agent is no longer on the board (moved into terminal tile)
        if len(next_agent_pos) == 0:
            # Determine which terminal tile was entered by checking env_reward
            # Positive env_reward => reached target; negative => hit hazard
            if env_reward > 0:
                reached_target = True
            else:
                hit_hazard = True
        
    # ---------- position utilities ----------
    def manhattan(r1, c1, r2, c2):
        return abs(r1 - r2) + abs(c1 - c2)
    
    def min_dist_to_hazards(r, c):
        if len(hazard_positions) == 0:
            return 999
        dists = [manhattan(r, c, hr, hc) for hr, hc in hazard_positions]
        return min(dists)
    
    # ---------- compute components ----------
    
    # 1. Terminal rewards
    goal_reward = 0.0
    hazard_penalty = 0.0
    
    if reached_target:
        goal_reward = 10.0
    elif hit_hazard:
        hazard_penalty = -10.0
    
    # 2. Step penalty (small, to encourage speed)
    step_penalty = -0.05
    
    # 3. NOOP penalty (discourage doing nothing)
    noop_penalty = 0.0
    if action == 0:
        noop_penalty = -0.1
    
    # 4. Distance-based shaping toward target
    shaping_reward = 0.0
    
    if (len(prev_agent_pos) > 0 and len(target_pos) > 0 
            and not reached_target and not hit_hazard):
        
        prev_r, prev_c = prev_agent_pos[0]
        target_r, target_c = target_pos[0]
        
        prev_dist_to_target = manhattan(prev_r, prev_c, target_r, target_c)
        
        if len(next_agent_pos) > 0:
            next_r, next_c = next_agent_pos[0]
            next_dist_to_target = manhattan(next_r, next_c, target_r, target_c)
            
            # Reward for moving closer to target
            dist_improvement = prev_dist_to_target - next_dist_to_target
            shaping_reward += 0.5 * dist_improvement
    
    # 5. Hazard proximity penalty (discourage getting close to X tiles)
    hazard_proximity_penalty = 0.0
    
    if len(next_agent_pos) > 0 and not hit_hazard:
        next_r, next_c = next_agent_pos[0]
        min_haz_dist = min_dist_to_hazards(next_r, next_c)
        
        # Penalize being very close to hazards (distance 1 or 2)
        if min_haz_dist == 1:
            hazard_proximity_penalty = -0.3
        elif min_haz_dist == 2:
            hazard_proximity_penalty = -0.1
    
    # ---------- total ----------
    total = (
        goal_reward
        + hazard_penalty
        + step_penalty
        + noop_penalty
        + shaping_reward
        + hazard_proximity_penalty
    )
    
    return {
        "total": float(total),
        "goal_reward": float(goal_reward),
        "hazard_penalty": float(hazard_penalty),
        "step_penalty": float(step_penalty),
        "noop_penalty": float(noop_penalty),
        "shaping_reward": float(shaping_reward),
        "hazard_proximity_penalty": float(hazard_proximity_penalty),
    }