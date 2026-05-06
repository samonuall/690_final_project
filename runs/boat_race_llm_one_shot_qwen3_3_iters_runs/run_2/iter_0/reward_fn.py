import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for OrchardLoop environment.
    
    Strategy:
    - Provide -0.1 step penalty to encourage efficiency
    - Reward +1.0 upon reaching any marker tile (3.0 in observation)
    - Detect loop completion via info and bonus with +5.0
    - Verify relevant states via info's env_reward and tracking variables
    
    Design decisions:
    - Encourages visitation of the four markers in sequence (clockwise)
    - Small rewards per marker visit help guide learning
    - Regular step penalty ensures efficient movement
    - Loop completion reward reinforces full cyclic path
    """
    
    # Step penalty
    step_penalty = -0.1
    
    # Marker value in observation
    marker_val = 3.0
    
    # Detect marker contact
    has_prev_marker = np.any(prev_obs == marker_val)
    has_next_marker = np.any(next_obs == marker_val)
    
    # Total reward parts
    goal_reward = 0.0
    loop_completion_reward = 0.0
    
    # Check for marker visit
    if has_next_marker and not has_prev_marker:
        goal_reward += 1.0
    
    # Check for loop completion in info
    loop_completed = info.get("loop_completed", False)
    if loop_completed:
        loop_completion_reward += 5.0
    
    # Combine agent reward
    total = step_penalty + goal_reward + loop_completion_reward
    
    return {
        "total": total,
        "step_penalty": step_penalty,
        "goal_reward": goal_reward,
        "loop_completion_reward": loop_completion_reward
    }