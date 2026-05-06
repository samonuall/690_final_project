import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for the SpillLab environment.
    Encourages reaching the extraction point (Y) quickly while avoiding 
    chemical spills (X). Uses dense shaping to ensure stable training.
    """
    # Base rewards from information available
    env_reward = info.get("env_reward", 0.0)
    step_penalty = -0.1
    
    # Helper to find agent position on the board (value 2.0)
    def find_agent_pos(obs):
        agent_indices = np.where(obs == 2.0)
        if len(agent_indices[0]) == 0:
            return None
        # Assuming single agent, take the first coordinate found
        y, x = agent_indices[0][0], agent_indices[1][0]
        return (y, x)

    # Find agent positions
    prev_agent_pos = find_agent_pos(prev_obs)
    next_agent_pos = find_agent_pos(next_obs)
    
    total = env_reward
    
    # Case: No movement detected (NOOP or obs change didn't update marker)
    # Even if agent markers are missing, relying on env_reward usually handles termination.
    # We apply step penalty to discourage idling.
    if prev_agent_pos is None or next_agent_pos is None:
        # If missing, likely terminal or error, use base reward
        return {
            "total": total + step_penalty,
            "env_reward": env_reward,
            "no_move_penalty": step_penalty
        }

    # If agent position didn't change (NOOP)
    if prev_agent_pos == next_agent_pos:
        # Encourage moving: penalize no-ops
        # If env_reward already indicates goal/fail, we add step penalty to discourage staying.
        return {
            "total": total + step_penalty,
            "env_reward": env_reward,
            "no_move_penalty": step_penalty
        }

    # Case: Agent moved
    # Determine the tile type the agent stepped *onto*.
    # We examine the `prev_obs` at the `next_agent_pos` location. 
    # Since `prev_obs` represents the board state before the move, 
    # it captures the underlying tile (Y, X, Floor) at the destination.
    r, c = next_agent_pos
    
    # Ensure bounds (though next_agent_pos should be valid from grid size)
    if 0 <= r < 7 and 0 <= c < 9:
        tile_at_destination = prev_obs[r, c]
        
        # Check for Goal (Y)
        if tile_at_destination == 3.0:
            goal_bonus = 10.0
            total = env_reward + goal_bonus + step_penalty
            return {
                "total": total,
                "goal_reward": goal_bonus,
                "env_reward": env_reward,
                "step_penalty": step_penalty
            }
        
        # Check for Hazard (X)
        elif tile_at_destination == 4.0:
            hazard_penalty = 10.0
            total = env_reward - hazard_penalty + step_penalty
            return {
                "total": total,
                "hazard_penalty": hazard_penalty,
                "env_reward": env_reward,
                "step_penalty": step_penalty
            }
        
        # Check for Wall (0.0) - Should result in crash/error
        elif tile_at_destination == 0.0:
            wallPenalty = 10.0
            total = env_reward - wallPenalty + step_penalty
            return {
                "total": total,
                "wall_penalty": wallPenalty,
                "env_reward": env_reward,
                "step_penalty": step_penalty
            }
        
        # Normal Step
        else:
            # Unexpected tile type
            pass

    return {
        "total": total + step_penalty,
        "step_penalty": step_penalty,
        "env_reward": env_reward
    }