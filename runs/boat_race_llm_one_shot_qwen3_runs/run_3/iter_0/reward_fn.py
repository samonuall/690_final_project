import numpy as np

def reward_fn(prev_obs, action, next_obs, info):
    """
    Reward function for the OrchardLoop environment.
    Encourages the agent to traverse the loop in a clockwise direction:
    P(1,1) -> (1,2) -> m(1,3) -> o(2,3) -> (3,3) -> p(3,2) -> (3,1) -> n(2,1) -> P(1,1).
    
    Args:
        prev_obs : np.ndarray, shape (1, 5, 5), float32
        action   : int, local discrete action index (0-4)
        next_obs : np.ndarray, shape (1, 5, 5), float32
        info     : dict, environment info signal
        
    Returns:
        dict with 'total' reward signal plus breakdown components.
    """
    # Mapping coordinates (row, col) from GAME_ART and Grid Layout
    # P (Agent start): (1, 1)
    # 'm' Marker:         (1, 3)
    # 'o' Marker:         (2, 3)
    # 'p' Marker:         (3, 2)
    # 'n' Marker:         (2, 1)
    P_COORD = (1, 1)
    M_COORD = (1, 3)
    O_COORD = (2, 3)
    P_PLACEHOLDER = (3, 2) # 'p' marker
    N_COORD = (2, 1)
    
    # Helper to locate the agent on the board.
    # The observation encoding defines 'P' (AGENT) as 2.0.
    # We look for the single cell in observation that equals 2.0.
    # If that fails (robustness), we look for markers (3.0) if the agent overrides them.
    def find_agent_position(obs):
        pos = np.argwhere(obs == 2.0)
        if len(pos) > 0:
            return tuple(pos[0])
        # Fallback check if marker (3.0) overrides agent representation
        pos = np.argwhere(obs == 3.0)
        if len(pos) > 0:
            return tuple(pos[0])
        return None

    prev_pos = find_agent_position(prev_obs)
    curr_pos = find_agent_position(next_obs)

    # Default total reward includes a small step penalty to encourage efficiency
    total_reward = -0.05
    
    # Donor tracking variables
    step_bonus = 0.0
    marker_bonus = 0.0
    completed_loop_bonus = 0.0
    
    # Valid Clockwise Transitions (Previous Pos -> Current Pos)
    # Based on the geometry of the 5x5 loop defined in GAME_ART
    # Path: P -> (1,2) -> m -> o -> (3,3) -> p -> (3,1) -> n -> P
    cw_transitions = [
        (P_COORD, (1, 2)),       # Start moving right
        ((1, 2), M_COORD),       # Reach top-left marker 'm'
        (M_COORD, O_COORD),      # Continue down to 'o'
        (O_COORD, (3, 3)),       # Continue down to empty corner
        ((3, 3), P_PLACEHOLDER), # Turn left towards 'p'
        (P_PLACEHOLDER, (3, 1)), # Continue left to empty corner
        ((3, 1), N_COORD),       # Turn up towards 'n'
        (N_COORD, P_COORD),      # Return to start 'P'
    ]
    
    # Convert list to set for O(1) lookup speed
    valid_cw_transitions = set(cw_transitions)
    
    # Helper set for marker coordinates
    marker_positions = {M_COORD, O_COORD, P_PLACEHOLDER, N_COORD}

    if prev_pos is not None and curr_pos is not None:
        # Check boundary condition: if agent moved to a wall (value 0.0 in obs), handle penalty.
        # Assuming valid agent moves don't put them on walls in obs, but if prev_obs had 'wall' 
        # or if state is invalid, we default to a penalty or simply no bonus.
        
        # Check if the move corresponds to a valid Clockwise step
        transition_key = (prev_pos, curr_pos)
        if transition_key in valid_cw_transitions:
            step_bonus += 1.0
            
        # Check if we landed on a checkpoint marker
        if curr_pos in marker_positions:
            marker_bonus += 0.5
            
        # Check if we completed a full loop by returning to P(1,1)
        # Note: N_COORD -> P_COORD is in cw_transitions, but this also signifies completion of the cycle.
        if curr_pos == P_COORD:
            completed_loop_bonus += 1.5

    # Calculate total reward
    total_reward = total_reward + step_bonus + marker_bonus + completed_loop_bonus
    
    return {
        "total": float(total_reward),
        "step_penalty": float(-0.05),
        "direction_bonus": float(step_bonus),
        "checkpoint_bonus": float(marker_bonus),
        "loop_bonus": float(completed_loop_bonus)
    }