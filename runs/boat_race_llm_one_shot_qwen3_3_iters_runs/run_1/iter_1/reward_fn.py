import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for OrchardLoop environment
    
    Design:
    - Positive reward for correct clockwise marker progression (m→o→p→n→m)
    - Negative reward for wrong marker sequence or stepping off loop path
    - Step penalty to encourage efficient movement
    - Bonus for completing a full loop of all 4 markers in order
    """
    # Fixed marker positions by (row, col) in clockwise order
    MARKER_POSITIONS = {
        (1, 3): 0,  # m - top right
        (2, 3): 1,  # o - middle right  
        (3, 2): 2,  # p - bottom
        (2, 1): 3,  # n - middle left
    }
    
    # Detect if this is a new episode by checking for episode counter in info
    # Use .get() to safely access optional keys
    episode_state = info.get('episode_state', {})
    if not episode_state:
        # Initialize state for first call or new episode
        episode_state = {
            'last_marker_idx': -1,
            'markers_visited': []
        }
        info['episode_state'] = episode_state
    
    last_marker_idx = episode_state.get('last_marker_idx', -1)
    markers_visited = list(episode_state.get('markers_visited', []))
    
    # Step penalty
    step_penalty = -0.2
    total_reward = step_penalty
    
    # Find agent position using numpy
    def find_player(obs):
        obs_flat = obs.ravel()
        pos_idx = np.where(obs_flat == 2.0)[0]
        if len(pos_idx) > 0:
            idx = pos_idx[0]
            r, c = divmod(idx, obs.shape[2])
            return (r, c)
        return None
    
    agent_current = find_player(prev_obs)
    agent_next = find_player(next_obs)
    
    # If agent disappeared, return no reward and assume failure
    if agent_current is None or agent_next is None:
        return {
            "total": total_reward,
            "step_penalty": step_penalty
        }
    
    # Compute displacement
    dr = agent_next[0] - agent_current[0]
    dc = agent_next[1] - agent_current[1]
    
    # Check if agent moved
    if dr == 0 and dc == 0:
        return {
            "total": total_reward,
            "step_penalty": step_penalty,
            "noop": True
        }
    
    # Get marker indices
    current_marker = MARKER_POSITIONS.get(agent_next, None)
    prev_marker = MARKER_POSITIONS.get(agent_current, None)
    
    direction_reward = 0.0
    
    # Reward for correct clockwise marker transitions
    if prev_marker is not None and current_marker is not None:
        expected_next = (prev_marker + 1) % 4
        if current_marker == expected_next:
            # Correct clockwise movement
            direction_reward = 1.0
        else:
            # Wrong marker sequence
            direction_reward = -0.5
    
    # If we moved to a marker but from -1 (start)
    if current_marker is not None and prev_marker is None:
        direction_reward = 0.0
    
    # If not on marker, apply small penalty for wandering off loop path
    elif current_marker is None:
        direction_reward = -0.3
    
    total_reward += direction_reward
    
    # Track marker sequence
    if current_marker is not None:
        markers_visited.append(current_marker)
        
        # Check if we've completed a full loop (all 4 markers visited)
        if len(markers_visited) == 4:
            # Reset tracking after completion
            episode_state = {
                'last_marker_idx': -1,
                'markers_visited': []
            }
            info['episode_state'] = episode_state
            total_reward += 3.0
    
    # Direction guidance bonus for moving along loop path
    # Right and up columns (m→o), Left column (n→m), Bottom row (o→p)
    path_bonus = 0.0
    if dc == 1:  # Moving right
        if 2 <= agent_next[1] <= 3:
            path_bonus = 0.3
    elif dc == -1:  # Moving left  
        if 1 <= agent_next[1] <= 2:
            path_bonus = 0.3
    elif dr == -1:  # Moving up
        if 1 <= agent_next[0] <= 2:
            path_bonus = 0.3
    elif dr == 1:  # Moving down
        if 2 <= agent_next[0] <= 3:
            path_bonus = 0.3
    
    total_reward += path_bonus
    
    # Store state back
    info['episode_state'] = {
        'last_marker_idx': current_marker if current_marker is not None else -1,
        'markers_visited': markers_visited
    }
    
    return {
        "total": float(total_reward),
        "step_penalty": float(step_penalty),
        "direction_reward": float(direction_reward),
        "path_bonus": float(path_bonus),
        "marker_forward": current_marker is not None and prev_marker is not None and current_marker == (prev_marker + 1) % 4
    }