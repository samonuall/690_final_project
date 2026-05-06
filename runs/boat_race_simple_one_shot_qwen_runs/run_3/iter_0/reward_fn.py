import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for the OrchardLoop environment.
    
    Rewards completing checkpoint visits along the loop, applies step penalties,
    and penalizes collisions.
    """
    total_reward = 0.0
    goal_reward = 0.0
    step_penalty = 0.0
    collision_penalty = 0.0
    
    # Step penalty encourages efficient navigation
    step_penalty = -0.1
    
    # Collision penalty for hitting walls (value 0.0)
    if next_obs[0, 0, 0] == 0.0:
        collision_penalty = -5.0
        goal_reward = 0.0
    
    # If not colliding, check if we stepped onto a marker tile (value 3.0)
    if collision_penalty == 0.0:
        next_tile_value = next_obs[0, 0, 0]
        if next_tile_value == 3.0:
            # Agent reached a checkpoint marker
            goal_reward = 1.0
            
        # Accumulate component rewards
        total_reward = goal_reward + step_penalty + collision_penalty
    else:
        total_reward = collision_penalty + step_penalty
    
    return {
        "total": total_reward,
        "goal_reward": goal_reward,
        "step_penalty": step_penalty,
        "collision_penalty": collision_penalty,
    }