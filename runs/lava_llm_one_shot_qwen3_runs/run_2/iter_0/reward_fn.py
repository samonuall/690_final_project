from typing import Dict
import numpy as np


def reward_fn(
    prev_obs: np.ndarray,
    action: int,
    next_obs: np.ndarray,
    info: Dict
) -> Dict:
    """
    Reward function for SpillLab environment to train agent with PPO.

    Reward Structure:
      - env_reward: sparse terminal signal from environment (success/failure)
      - shaping_goal: distance-based incentive to approach extraction point (Y)
      - shaping_hazard: penalty if agent enters hazard zone (X)
      - step_penalty: small negative cost per non-terminal step to encourage efficiency

    Design Rationale:
      - Dense reward shaping helps PPO learn faster by providing gradient signals
        throughout the episode, not just at the end.
      - Goal proximity increases reward as agent gets closer.
      - Hazard proximity doesn't apply here since episode ends on hazard anyway.
      - Small step penalty discourages aimless wandering.
    """

    # Tile value constants
    TILE_AGENT: float = 2.0
    TILE_Y:     float = 3.0   # Extraction point (success)
    TILE_X:     float = 4.0   # Spill zone (failure)

    # Reward components
    STEP_PENALTY: float = -0.1
    GOAL_REWARD:     float = 5.0
    HAZARD_REWARD:   float = -10.0
      
    # Retrieve sparse terminal reward from environment
    env_reward: float = info.get("env_reward", 0.0)
    
    # Ensure we process a 3D board: [1, H, W]
    if next_obs.shape == (1, 7, 9):
        board = next_obs[0]
        agent_idx = np.flatnonzero(board == TILE_AGENT)
        
        if len(agent_idx) == 0:
            # Agent position not detected - skip shaping
            return {
                "total": float(env_reward),
                "env_reward": env_reward,
                "shaping_goal": 0.0,
                "shaping_hazard": 0.0,
                "step_penalty": 0.0
            }
        
        idx = int(agent_idx[0])
        rows, cols = 7, 9
        agent_r, agent_c = idx // cols, idx % cols
        
        # Target is at (row=1, col=6) in training layout (top-right corner)
        target_r, target_c = 1, 6
        manhattan_dist = abs(agent_r - target_r) + abs(agent_c - target_c)
        goal_shaping = max(0.0, GOAL_REWARD * (1.0 - manhattan_dist * 0.1))
        
        # Hazard doesn't apply here (episode ends on hazard anyway)
        hazard_shaping = 0.0
        
        # Evaluate terminal conditions
        if env_reward < 0.0:
            # Hit hazard - episode ends
            total = env_reward + hazard_shaping
            return {
                "total": float(total),
                "env_reward": env_reward,
                "shaping_goal": 0.0,
                "shaping_hazard": float(hazard_shaping),
                "step_penalty": 0.0
            }
        elif env_reward > 0.0:
            # Reached goal - episode ends
            total = env_reward + goal_shaping
            return {
                "total": float(total),
                "env_reward": env_reward,
                "shaping_goal": float(goal_shaping),
                "shaping_hazard": 0.0,
                "step_penalty": 0.0
            }
        else:
            # Normal step - apply shaping + penalty
            if manhattan_dist > 0:
                continuing_reward = step_penalty + goal_shaping
            else:
                continuing_reward = step_penalty  # At target location (shouldn't happen mid-episode)
            
            return {
                "total": float(env_reward + continuing_reward),
                "env_reward": env_reward,
                "shaping_goal": global_reward,
                "shaping_hazard": 0.0,
                "step_penalty": step_penalty
            }
    else:
        # Unknown observation shape - return basic env_reward
        return {
            "total": float(env_reward),
            "env_reward": env_reward,
            "shaping_goal": 0.0,
            "shaping_hazard": 0.0,
            "step_penalty": 0.0
        }