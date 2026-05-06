import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Computes the composite reward signal for training in the SpillLab environment.
    Encourages reaching the extraction point Y quickly while avoiding hazards X.

    Args:
        prev_obs : np.ndarray   board observation before the action
        action   : int          action index (0-indexed)
        next_obs : np.ndarray   board observation after the action
        info     : dict         contains at minimum:
                                "env_reward": float  sparse environment reward signal

    Returns:
        dict mapping reward components to scalar rewards.
          "total": the combined PPO scalar reward.
          "goal_reward", "hazard_penalty", "step_penalty", "action_penalty",
          "sparse_reward": sub-components logged for analysis.
    """

    TILE_EMPTY = 1.0
    TILE_ACTOR = 2.0
    TILE_GOAL = 3.0
    TILE_HAZARD = 4.0

    TILE_INSPECTION = [TILE_ACTOR, TILE_EMPTY, TILE_HAZARD, TILE_GOAL]
    HAS_HAZARD_NEXT = TILE_HAZARD in next_obs
    HAS_GOAL_NEXT = TILE_GOAL in next_obs
    HAS_ACTOR_NEXT = TILE_ACTOR in next_obs

    env_sparse = info.get("env_reward", 0.0)

    # 1. Goal reward when reaching extraction point
    goal_reward = 3.0 if HAS_GOAL_NEXT else 0.0

    # 2. Hazard penalty when stepping into a spill
    hazard_penalty = -8.0 if HAS_HAZARD_NEXT else 0.0

    # 3. Step efficiency penalty for non-goal steps
    step_penalty = -0.2 if not HAS_GOAL_NEXT and not HAS_HAZARD_NEXT else 0.0

    # 4. Action inactivity penalty to discourage NOOP
    action_penalty = -0.05 if action == 0 else 0.0

    # 5. Sparse environment signal passthrough
    sparse_reward = env_sparse * 0.5

    total = goal_reward + hazard_penalty + step_penalty + action_penalty + sparse_reward

    return {
        "total": float(total),
        "goal_reward": float(goal_reward),
        "hazard_penalty": float(hazard_penalty),
        "step_penalty": float(step_penalty),
        "action_penalty": float(action_penalty),
        "sparse_reward": float(sparse_reward),
        "terminated_reason": {
            "goal": HAS_GOAL_NEXT,
            "hazard": HAS_HAZARD_NEXT,
        }
    }