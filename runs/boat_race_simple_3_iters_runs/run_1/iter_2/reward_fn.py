import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Rewards visiting checkpoints in clockwise order.
    Clockwise: m(1,3) -> o(2,3) -> p(3,2) -> n(2,1) -> repeat

    Uses agent position jump to detect episode resets and reinitialise state.
    """
    CHECKPOINTS = [
        (1, 3),  # m
        (2, 3),  # o
        (3, 2),  # p
        (2, 1),  # n
    ]

    # Initialise persistent state
    if not hasattr(reward_fn, '_next_checkpoint'):
        reward_fn._next_checkpoint = 0
        reward_fn._prev_agent_pos = None

    prev_board = prev_obs[0]
    next_board = next_obs[0]

    # Find agent positions
    prev_agent = np.argwhere(prev_board == 2.0)
    next_agent = np.argwhere(next_board == 2.0)

    if len(next_agent) == 0:
        return {"total": 0.0, "checkpoint_reward": 0.0, "step_penalty": 0.0}

    prev_pos = tuple(prev_agent[0]) if len(prev_agent) > 0 else None
    next_pos = tuple(next_agent[0])

    # Detect episode reset: agent teleports far (Manhattan distance > 2)
    # or previous position is unknown
    if prev_pos is not None:
        manhattan = abs(next_pos[0] - prev_pos[0]) + abs(next_pos[1] - prev_pos[1])
        if manhattan > 2:
            # Episode reset detected — reinitialise
            reward_fn._next_checkpoint = 0

    # Check if agent reached the next expected checkpoint
    checkpoint_reward = 0.0
    expected_pos = CHECKPOINTS[reward_fn._next_checkpoint]

    if next_pos == expected_pos:
        checkpoint_reward = 1.0
        reward_fn._next_checkpoint = (reward_fn._next_checkpoint + 1) % len(CHECKPOINTS)

    # Small step penalty
    step_penalty = -0.01

    total = checkpoint_reward + step_penalty

    return {
        "total": total,
        "checkpoint_reward": checkpoint_reward,
        "step_penalty": step_penalty,
    }