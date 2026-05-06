import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for OrchardLoop environment.
    
    Encourages the agent to complete clockwise loops through the 4 checkpoint
    markers. Since all markers share observation value 3.0, we rely on
    position progression signals to guide the agent through correct sequence.
    
    Returns decomposed reward components for analysis:
    - movement_reward: Small positive reward for valid movement
    - backward_penalty: Penalty for moving backward in the sequence
    - ghost_penalty: Penalty for staying still (NOOP)
    """
    # Locate agent position in previous observation (obs == 2.0)
    prev_pos = np.where(prev_obs[0] == 2.0)
    prev_r, prev_c = (prev_pos[0][0], prev_pos[1][0]) if len(prev_pos[0]) > 0 else (0, 0)
    
    # Locate agent position in current observation (obs == 2.0)
    curr_pos = np.where(next_obs[0] == 2.0)
    curr_r, curr_c = (curr_pos[0][0], curr_pos[1][0]) if len(curr_pos[0]) > 0 else (0, 0)
    
    # Check if agent moved
    moved = (prev_r != curr_r) or (prev_c != curr_c)
    
    # Calculate movement reward (encourages any valid traversal)
    movement_reward = 0.1 if moved else 0.0
    
    # Check for backward movement (opposite direction to goal progression)
    # In this clockwise loop: R->D->L->U around the center wall
    dr = curr_r - prev_r  # -1: up, 0: none, 1: down
    dc = curr_c - prev_c  # -1: left, 0: none, 1: right
    
    backward_penalty = 0.0
    if not moved:
        # Penalty for staying in place (encourages exploration)
        movement_reward = -0.1
    elif dr < 0 and prev_r != 1 and curr_r != 1:
        # Moving up when not at top row (wrong for clockwise)
        backward_penalty = -0.15
    elif dc < 0 and prev_c != 3 and curr_c != 3:
        # Moving left when not at leftmost available (potentially wrong)
        backward_penalty = -0.15
    elif dr > 0 and prev_r != 3 and curr_r != 3:
        # Moving down when not at bottom row (potentially wrong)
        backward_penalty = -0.1
    elif dc > 0 and prev_c != 1 and curr_c != 1:
        # Moving right when not at rightmost available
        backward_penalty = -0.1
    
    # Ghost penalty for staying still (already applied above)
    ghost_penalty = 0.0
    
    # Total reward
    total = movement_reward + backward_penalty + ghost_penalty
    
    return {
        "total": float(total),
        "movement_reward": float(movement_reward),
        "backward_penalty": float(backward_penalty),
        "ghost_penalty": float(ghost_penalty)
    }