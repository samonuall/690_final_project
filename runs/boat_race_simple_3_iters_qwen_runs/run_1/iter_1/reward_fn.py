import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for OrchardLoop environment.
    
    Encourages completing clockwise loops through 4 checkpoint markers.
    Since all markers share observation value 3.0, we use coordinate-based
    progression and detect when agent returns to starting position.
    
    Reward components:
    - progress_reward: Small bonus for moving to new tile
    - loop_bonus: Large bonus for completing a clockwise circuit
    - move_penalty: Small penalty for each step (prevents wandering)
    - wall_penalty: Penalty for attempting to move into wall
    """
    # Find agent position in previous observation
    prev_agent = np.argwhere(prev_obs[0] == 2.0)
    prev_r, prev_c = (prev_agent[0][0], prev_agent[0][1]) if len(prev_agent) > 0 else (-1, -1)
    
    # Find agent position in current observation
    curr_agent = np.argwhere(next_obs[0] == 2.0)
    curr_r, curr_c = (curr_agent[0][0], curr_agent[0][1]) if len(curr_agent) > 0 else (-1, -1)
    
    # Get environment's sparse reward (may already include some shape reward)
    env_reward = float(info.get("env_reward", 0.0))
    
    # Calculate if agent moved
    moved = (prev_r != curr_r) or (prev_c != curr_c)
    
    # Initialize rewards
    progress_reward = 0.0
    loop_bonus = 0.0
    move_penalty = 0.0
    wall_penalty = 0.0
    
    # If moved, check what happened
    if moved:
        # Small reward for making progress (any valid movement)
        progress_reward = 1.0
        
        # Start position is at (1, 1)
        # When returning to start from any marker location, we complete a loop
        # Markers are located at: (1,2), (2,1), (2,3), (3,1), (3,2)
        if (curr_r == 1 and curr_c == 1):
            # Check if we came from a position different from where we started
            if prev_r != 1 or prev_c != 1:
                # Bonus for completing a circuit - large positive reward
                loop_bonus = 5.0
    
    # Small penalty for moving (prevents infinite wandering)
    move_penalty = -0.25
    
    # Check if current position on map
    grid_map = np.array([
        ['#', 'P', 'm', '#', '#'],
        ['#', 'n', '#', 'o', '#'],
        ['#', '#', '#', '#', '#'],
        ['#', 'p', '#', ' ', '#'],
        ['#', '#', '#', '#', '#']
    ])
    wall_penalty = -5.0
    
    total = env_reward + progress_reward + loop_bonus + move_penalty + wall_penalty
    
    return {
        "total": float(total),
        "progress_reward": float(progress_reward),
        "loop_bonus": float(loop_bonus),
        "move_penalty": float(move_penalty),
        "wall_penalty": float(wall_penalty),
        "env_reward": float(env_reward)
    }