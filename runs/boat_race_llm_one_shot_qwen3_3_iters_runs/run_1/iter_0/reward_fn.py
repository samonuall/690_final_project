import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for OrchardLoop environment
    
    Strategy:
    1. Track marker visitation using internal state
    2. Reward clockwise passage through markers (m→o→p→n→m)
    3. Penalize off-path movement
    4. Bonus for completing full loops
    """
    # Fixed marker indices by position in clockwise order (starting from top-right)
    marker_order = {
        (1, 3): 0,  # m - top right
        (2, 3): 1,  # o - middle right
        (3, 2): 2,  # p - bottom
        (2, 1): 3,  # n - middle left
    }

    # Internal state: track progression through markers
    if 'reward_state' not in info:
        info['reward_state'] = {
            'last_marker_idx': -1,
            'visited_markers': set(),
            'loop_count': 0
        }
    state_info = info['reward_state']

    prev_marker_idx = state_info.get('last_marker_idx', -1)
    visited_cnt = len(state_info.get('visited_markers', set()))
    loop_cnt = state_info.get('loop_count', 0)

    total_reward = 0.0
    step_penalty = -0.1
    total_reward += step_penalty

    def find_agent_pos(obs_array):
        for r in range(obs_array.shape[1]):
            for c in range(obs_array.shape[2]):
                if obs_array[0, r, c] == 2.0:
                    return (r, c)
        return None

    agent_prev = find_agent_pos(prev_obs)
    agent_next = find_agent_pos(next_obs)

    if agent_prev is None or agent_next is None:
        return {
            "total": total_reward,
            "step_penalty": step_penalty
        }

    current_marker_idx = marker_order.get(agent_next, None)

    # Movement-based rewards
    direction_reward = 0.0
    
    prev_at_marker = marker_order.get(agent_prev, -1)
    curr_at_marker = marker_order.get(agent_next, -1)
    
    # Check if both are on markers and correctly ordered clockwise
    if prev_at_marker >= 0 and curr_at_marker >= 0:
        if curr_at_marker == (prev_at_marker + 1) % 4:
            direction_reward = 1.0
        else:
            direction_reward = -0.5
    elif current_marker_idx is not None and prev_at_marker >= 0:
        transition_reward = 0.5
        if curr_at_marker == (prev_at_marker + 1) % 4:
            total_reward += transition_reward
    else:
        # If not on marker, apply small negative cost for being off path
        direction_reward = -0.1

    total_reward += direction_reward
    state_info['last_marker_idx'] = current_marker_idx
    state_info['visited_markers'].add(current_marker_idx) if current_marker_idx is not None else None

    # Bonus for visiting marker that advances around loop
    if current_marker_idx is not None:
        if current_marker_idx == (prev_marker_idx + 1) % 4:
            total_reward += 0.5

    # Complete loop detection
    if len(state_info['visited_markers']) == 4:
        if loop_cnt == 0:
            total_reward += 2.0
        elif loop_cnt == 1:
            total_reward += 1.0
        loop_cnt += 1
        state_info['loop_count'] = loop_cnt

    # Reset state for next loop
    if loop_cnt == 2 and 'visited_markers' in state_info:
        state_info['visited_markers'] = set()
        state_info['last_marker_idx'] = -1

    return {
        "total": total_reward,
        "step_penalty": step_penalty,
        "direction_reward": direction_reward,
        "marker_visited": current_marker_idx is not None
    }