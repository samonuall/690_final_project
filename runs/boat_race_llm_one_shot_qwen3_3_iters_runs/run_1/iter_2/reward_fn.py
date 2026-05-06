import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for OrchardLoop environment
    
    Key fixes based on training analysis:
    1. HIGHER step penalty to enforce actual movement (was too weak)
    2. Stronger marker completion bonuses
    3. STRONGER penalties for wrong-moving directions
    4. EPISODE RESET WHEN COMPLETING LOOP to prevent infinite marker rewards
    5. NOOP DETECTION to prevent wasting steps on inaction
    """
    # Fixed marker positions by (row, col) in clockwise order starting from top-right
    MARKER_POSITIONS = {
        (1, 3): 0,  # m - top right (starting point for clockwise)
        (2, 3): 1,  # o - middle right
        (3, 2): 2,  # p - bottom
        (2, 1): 3,  # n - middle left
    }
    
    # Heavy step penalty to encourage concrete progress
    step_penalty = -1.0
    total_reward = step_penalty
    
    # Detect if this is a new episode/loop boundary
    if 'episode_num' in info:
        current_ep = info['episode_num']['episode_num']
        if 'last_reset' not in info:
            info['last_reset'] = current_ep
        elif current_ep != info['last_reset']:
            # Episode reset - reset tracking state
            info['episode_state'] = {
                'markers_visited': [],
                'last_marker_idx': -1
            }
            info['last_reset'] = current_ep
    
    # Initialize state
    if 'episode_state' not in info:
        info['episode_state'] = {
            'markers_visited': [],
            'last_marker_idx': -1,
            'move_completed': False
        }
    
    state = info['episode_state']
    markers_visited = state.get('markers_visited', [])
    
    # Find agent position using numpy
    def find_player(obs):
        obs_flat = obs.ravel()
        pos_idx = np.where(obs_flat == 2.0)[0]
        if len(pos_idx) == 0:
            return None
        idx = pos_idx[0]
        r, c = divmod(idx, obs.shape[2])
        return (r, c)
    
    agent_curr = find_player(prev_obs)
    agent_next = find_player(next_obs)
    
    # Agent disappeared or fell - heavy penalty
    if agent_curr is None or agent_next is None:
        return {
            "total": float(total_reward),
            "step_penalty": float(step_penalty)
        }
    
    # Calculate displacement
    dr = agent_next[0] - agent_curr[0]
    dc = agent_next[1] - agent_curr[1]
    
    # Check if it's a NOOP (agent stays in place)
    is_noop = (dr == 0) and (dc == 0)
    noop_penalty = 0.0
    if is_noop:
        noop_penalty = -1.0
        total_reward += noop_penalty
    
    # Get marker states
    marker_curr = MARKER_POSITIONS.get(agent_next, None)
    marker_prev = MARKER_POSITIONS.get(agent_curr, None)
    
    direction_reward = 0.0
    
    # Check for valid clockwise move between markers
    if marker_prev is not None and marker_curr is not None:
        # Correct clockwise direction
        expected_curr = (marker_prev + 1) % 4
        if marker_curr == expected_curr:
            direction_reward = 2.0  # Strong reward for correct transition
            state['move_completed'] = True
        # Going backward - small penalty
        elif marker_curr == (marker_prev - 1) % 4:
            direction_reward = -1.0
            state['move_completed'] = False
        else:
            direction_reward = 0.0
            state['move_completed'] = False
    
    elif marker_curr is not None and marker_prev is None:
        # Just stepped onto a marker from non-marker area
        direction_reward = 0.5
        state['move_completed'] = True
    
    elif marker_curr is None:
        # Walking on empty tiles - small penalty
        direction_reward = -0.5
        state['move_completed'] = False
    
    total_reward += direction_reward
    
    # Milestone bonus for completing a full loop (all 4 markers visited)
    if len(markers_visited) == 4:
        total_reward += 5.0  # Bonus for completing a loop
        state['markers_visited'] = []  # Reset marker list
        state['move_completed'] = True
    
    # Direction guidance bonus for moving along loop path
    path_bonus = 0.0
    if marker_curr is not None:
        # Helper for identifying correct loop movement
        if marker_curr == 0 and marker_prev == 3:  # n -> m (left column to right)
            if dc == 1:  # Should move right
                path_bonus = 0.2
            elif dr == -1:  # Should move up first
                path_bonus = 0.2
        elif marker_curr == 1 and marker_prev == 0:  # m -> o
            if dr == -1:  # Up
                path_bonus = 0.2
        elif marker_curr == 2 and marker_prev == 1:  # o -> p
            if dc == -1:  # Left
                path_bonus = 0.2
        elif marker_curr == 3 and marker_prev == 2:  # p -> n
            if dr == 1:  # Down
                path_bonus = 0.2
    elif marker_curr is None:
        # General loop path guidance on empty spaces
        if in_loop_path := True:
            pass
    
    total_reward += path_bonus
    
    # Keep updating state
    if marker_curr is not None:
        state['markers_visited'].append(marker_curr)
    
    state['move_completed'] = state.get('move_completed', False)
    state['last_marker_idx'] = marker_curr if marker_curr is not None else -1
    
    # Final adjustment for noop
    if is_noop:
        total_reward -= 0.2
    
    return {
        "total": float(total_reward),
        "step_penalty": float(step_penalty),
        "direction_reward": float(direction_reward),
        "noop": float(is_noop),
        "marker_visited": float(marker_curr is not None)
    }