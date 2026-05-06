import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Rewards the agent for visiting checkpoints in clockwise order:
    m(row=1,col=3) -> o(row=2,col=3) -> p(row=3,col=2) -> n(row=2,col=1) -> repeat
    """
    # Clockwise checkpoint positions (row, col)
    CHECKPOINTS = [
        (1, 3),  # m
        (2, 3),  # o
        (3, 2),  # p
        (2, 1),  # n
    ]

    # Use a mutable default argument to persist state across calls
    if not hasattr(reward_fn, '_next_checkpoint'):
        reward_fn._next_checkpoint = 0

    board = next_obs[0]  # shape (5, 5)

    # Find agent position
    agent_positions = np.argwhere(board == 2.0)
    if len(agent_positions) == 0:
        return {"total": 0.0, "checkpoint_reward": 0.0, "step_penalty": 0.0}

    agent_pos = tuple(agent_positions[0])  # (row, col)

    # Check if agent is on the next expected checkpoint
    checkpoint_reward = 0.0
    next_idx = reward_fn._next_checkpoint
    expected_pos = CHECKPOINTS[next_idx]

    if agent_pos == expected_pos:
        checkpoint_reward = 1.0
        reward_fn._next_checkpoint = (next_idx + 1) % len(CHECKPOINTS)

    # Small step penalty to encourage efficiency
    step_penalty = -0.01

    # NOOP penalty to discourage standing still
    noop_penalty = -0.05 if action == 0 else 0.0

    total = checkpoint_reward + step_penalty + noop_penalty

    return {
        "total": total,
        "checkpoint_reward": checkpoint_reward,
        "step_penalty": step_penalty,
        "noop_penalty": noop_penalty,
    }