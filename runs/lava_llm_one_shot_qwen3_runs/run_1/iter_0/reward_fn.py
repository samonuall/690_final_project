import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for the SpillLab environment.

    This function designs a dense reward signal to train the RL agent to:
    - Reach the extraction point (Y) as quickly as possible (+10)
    - Avoid chemical spill zones (X) completely (-20)
    - Prefer efficient paths over wandering (-0.1 per step)

    Reward components:
    - goal_reward: +10.0 if agent lands on target extraction point (Y)
    - hazard_reward: -20.0 if agent lands on chemical spill (X)
    - step_penalty: -0.1 for each valid movement step
    - wall_collision: -1.0 if agent attempts to move into a wall (#)

    Args:
        prev_obs: np.ndarray shape (1, 7, 9) - board observation before the action
        action: int - action index (0-fixed directions)
        next_obs: np.ndarray shape (1, 7, 9) - board observation after the action
        info: dict - environment info, may contain "terminal" flag

    Returns:
        dict with "total" reward and component breakdown for analysis
    """
    goal_reward = 0.0
    hazard_reward = 0.0
    wall_collision = 0.0
    step_penalty = 0.0

    curr_data = next_obs[0, :]
    prev_data = prev_obs[0, :]

    # Check for state change (movement occurred)
    if not np.array_equal(curr_data, prev_data):
        # Check for hazard tile (X = 4.0) - termination with punishment
        if np.max(curr_data) == 4.0:
            hazard_reward = -20.0

        # Check for goal tile (Y = 3.0) - termination with large reward
        elif np.max(curr_data) == 3.0:
            goal_reward = 10.0

        # Check for wall collision (observing 0.0 where agent might collide)
        # This catches attempts to move from Α当前位置 into a wall
        elif np.min(curr_data) == 0.0 and np.max(curr_data) == 0.0:
            wall_collision = -1.0

        # Regular valid step on empty tile (1.0) or continued progression
        else:
            step_penalty = -0.0

    return {
        "total": float(goal_reward + hazard_reward + wall_collision + step_penalty),
        "goal_reward": float(goal_reward),
        "hazard_reward": float(hazard_reward),
        "wall_collision": float(wall_collision),
        "step_penalty": float(step_penalty)
    }