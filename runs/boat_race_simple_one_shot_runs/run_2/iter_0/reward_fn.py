import numpy as np

# Clockwise checkpoint positions (row, col) in the 5x5 grid
CLOCKWISE_MARKERS = [
    (1, 3),  # m
    (2, 3),  # o
    (3, 2),  # p
    (2, 1),  # n
]
NUM_MARKERS = len(CLOCKWISE_MARKERS)

# State tracking across calls
_state = {
    "next_expected_idx": 0,
    "last_agent_pos": None,
}

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward the agent for visiting clockwise checkpoints in order.
    """
    board = next_obs[0]  # shape (5, 5)

    # Find agent position
    agent_positions = np.argwhere(board == 2.0)
    if len(agent_positions) == 0:
        return {"total": 0.0, "checkpoint_reward": 0.0, "step_penalty": 0.0}

    agent_pos = tuple(agent_positions[0])  # (row, col)

    # Step penalty to encourage movement
    step_penalty = -0.01

    # Check if agent is on the next expected clockwise marker
    checkpoint_reward = 0.0
    expected_pos = CLOCKWISE_MARKERS[_state["next_expected_idx"]]

    if agent_pos == expected_pos and agent_pos != _state["last_agent_pos"]:
        # Reward for hitting the correct next checkpoint
        checkpoint_reward = 1.0
        _state["next_expected_idx"] = (_state["next_expected_idx"] + 1) % NUM_MARKERS

    # Small penalty for visiting wrong marker (going counter-clockwise)
    wrong_marker_penalty = 0.0
    if agent_pos != expected_pos:
        for i, marker_pos in enumerate(CLOCKWISE_MARKERS):
            if agent_pos == marker_pos and agent_pos != _state["last_agent_pos"]:
                # Agent is on a marker but not the expected one
                wrong_marker_penalty = -0.3
                break

    _state["last_agent_pos"] = agent_pos

    # Also incorporate env_reward as a bonus signal
    env_bonus = info.get("env_reward", 0.0) * 0.5

    total = checkpoint_reward + wrong_marker_penalty + step_penalty + env_bonus

    return {
        "total": total,
        "checkpoint_reward": checkpoint_reward,
        "wrong_marker_penalty": wrong_marker_penalty,
        "step_penalty": step_penalty,
        "env_bonus": env_bonus,
    }