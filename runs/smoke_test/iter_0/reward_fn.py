import numpy as np

# Clockwise marker positions: m -> o -> p -> n -> m -> ...
# Layout:
# #####
# #P m#   m at row=1, col=3
# #n#o#   n at row=2, col=1 ; o at row=2, col=3
# # p #   p at row=3, col=2
# #####
CLOCKWISE_MARKERS = [
    (1, 3),  # m
    (2, 3),  # o
    (3, 2),  # p
    (2, 1),  # n
]
NUM_MARKERS = len(CLOCKWISE_MARKERS)

# State tracking (persisted across calls via mutable dict)
_state = {
    "next_marker_idx": 0,   # which marker we expect next (clockwise)
    "loops_completed": 0,
    "prev_agent_pos": None,
}

def _get_agent_pos(obs):
    """Return (row, col) of agent (value 2.0) in the 5x5 grid."""
    board = obs[0]  # shape (5, 5)
    positions = np.argwhere(board == 2.0)
    if len(positions) == 0:
        return None
    return tuple(positions[0])

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward the agent for visiting clockwise markers in sequence.
    """
    # Base components
    step_penalty = -0.01
    checkpoint_reward = 0.0
    loop_bonus = 0.0

    agent_pos = _get_agent_pos(next_obs)

    if agent_pos is not None:
        # Check if agent is on the next expected clockwise marker
        expected_marker = CLOCKWISE_MARKERS[_state["next_marker_idx"]]

        if agent_pos == expected_marker:
            # Agent reached the next marker in clockwise order!
            checkpoint_reward = 1.0
            _state["next_marker_idx"] = (_state["next_marker_idx"] + 1) % NUM_MARKERS

            # Check if a full loop was just completed (wrapped back to index 0)
            if _state["next_marker_idx"] == 0:
                _state["loops_completed"] += 1
                loop_bonus = 2.0  # bonus for completing a full loop

    # Incorporate sparse env_reward signal if present
    env_reward = info.get("env_reward", 0.0)

    total = step_penalty + checkpoint_reward + loop_bonus + env_reward

    return {
        "total": total,
        "step_penalty": step_penalty,
        "checkpoint_reward": checkpoint_reward,
        "loop_bonus": loop_bonus,
        "env_reward": env_reward,
    }