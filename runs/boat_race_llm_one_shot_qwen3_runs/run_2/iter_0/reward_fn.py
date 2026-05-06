import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for OrchardLoop Environment.
    
    This function computes a reward signal to encourage the agent to:
    1. Visit markers in clockwise order (m -> o -> p -> n).
    2. Complete a loop back to the start (P).
    3. Avoid collisions with walls (including the center obstacle).
    
    It maintains a visited state in `info` to track the sequence of markers hit.
    """
    
    # Configuration for marker locations (row, col) - 0-indexed, grid 5x5
    MARKERS_COORDS = {
        'm': (1, 3),  # Top-Right
        'o': (2, 3),  # Right Edge
        'p': (3, 2),  # Bottom (Center-bottom)
        'n': (2, 1),  # Left Edge
    }
    START_POS = (1, 1)
    CENTER_WALL_POS = (2, 2)
    
    # Function to extract agent position from observation
    def get_agent_pos(obs):
        mask = obs[0] == 2.0
        indices = np.argwhere(mask)
        if indices.size == 0:
            return None
        # Convert numpy array to tuple (row, col)
        return tuple(indices[0])
    
    prev_pos = get_agent_pos(prev_obs)
    curr_pos = get_agent_pos(next_obs)
    
    # Safety check for invalid observations
    if prev_pos is None or curr_pos is None:
        return {"total": -0.5, "step_penalty": -0.5, "env_reward": 0.0}
    
    total_reward = 0.0
    step_penalty = -0.1  # Small penalty for taking steps
    goal_reward = 0.0
    collision_penalty = 0.0
    
    # 1. Collision Penalty Check
    if curr_pos in MARKERS_COORDS.values():
        # Cannot be in multiple positions, but handle wall collision
        pass
    
    # Check if agent moved to center wall (shouldn't happen if action is valid)
    if curr_pos == CENTER_WALL_POS:
        collision_penalty = -5.0
        total_reward += collision_penalty
    else:
        # Check if action led to a valid position (not causing a step to become a wall)
        neighbor_val = next_obs[0][curr_pos[0], curr_pos[1]]
        # Walls have value 0.0
        if neighbor_val == 0.0:
            collision_penalty = -5.0
            total_reward += collision_penalty
        else:
            # Normal movement
            pass
    
    # 2. Marker Reward with Sequence Logic
    if curr_pos in MARKERS_COORDS:
        marker_key = None
        for k, v in MARKERS_COORDS.items():
            if v == curr_pos:
                marker_key = k
                break
        
        if marker_key:
            # Markers appear once per visit in order: m -> o -> p -> n
            info_vis = info.get('visited_markers', [])
            
            # Check if we haven't already visited this marker
            if marker_key not in info_vis:
                expected_sequence = ['m', 'o', 'p', 'n']
                
                # Determine if this marker is the expected next in sequence
                should_reward = False
                
                if len(info_vis) == 0:
                    # First visit - only reward 'm'
                    if marker_key == 'm':
                        should_reward = True
                else:
                    # Check sequential order
                    last_visited_idx = expected_sequence.index(info_vis[-1])
                    next_idx = last_visited_idx + 1
                    if next_idx < 4 and marker_key == expected_sequence[next_idx]:
                        should_reward = True
                
                if should_reward:
                    goal_reward = 1.0  # Reward for visiting next marker in sequence
                    # Update visited list
                    info['visited_markers'] = info_vis + [marker_key]
                    total_reward += goal_reward
            
            # Check if last marker was completed (bonus)
            if info_vis == ['m', 'o', 'p', 'n']:
                goal_reward += 5.0  # Completion bonus
    else:
        # Not on a marker, just continue sequence tracking
        info_vis = info.get('visited_markers', [])
        # Reset tracking info if it wasn't initialized
        info['visited_markers'] = info_vis
    
    # 3. Loop Completion Reward (Return to Start with all markers visited)
    if curr_pos == START_POS:
        info_vis = info.get('visited_markers', [])
        if info_vis == ['m', 'o', 'p', 'n']:
            goal_reward += 10.0  # Large bonus for completing full loop
            info['visited_markers'] = []  # Reset for next loop
    
    # 4. Add step penalty for movement
    total_reward += step_penalty
    
    # 5. Add any existing sparse reward from env
    env_reward = info.get("env_reward", 0.0)
    total_reward += env_reward
    
    # Build return dict
    result = {
        "total": total_reward,
        "step_penalty": step_penalty,
        "env_reward": env_reward
    }
    
    if goal_reward != 0.0:
        result["goal_reward"] = goal_reward
    if collision_penalty != 0.0:
        result["collision_penalty"] = collision_penalty
    
    return result