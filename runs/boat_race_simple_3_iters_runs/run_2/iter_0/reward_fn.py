import numpy as np

# Marker positions (row, col) in the 5x5 grid
# Layout:
# #####
# #P m#
# #n#o#
# # p #
# #####
# m=(1,3), n=(2,1), o=(2,3), p=(3,2)
# Clockwise order: m -> o -> p -> n -> m -> ...
CLOCKWISE_MARKERS = [
    (1, 3),  # m
    (2, 3),  # o
    (3, 2),  # p
    (2, 1),  # n
]

# State tracking (persisted across calls via mutable default)
_state = {
    "last_marker_idx": None,   # index in CLOCKWISE_MARKERS of last visited marker
    "visited_positions": set(),# to detect revisits within a segment
}

def _get_agent_pos(obs):
    """Find agent position (row, col) in the 5x5 grid."""
    board = obs[0]  # shape (5, 5)
    positions = np.argwhere(board == 2.0)
    if len(positions) > 0:
        return (int(positions[0][0]), int(positions[0][1]))
    return None

def _is_marker_pos(pos):
    """Check if position is a marker and return its index, or -1."""
    if pos is None:
        return -1
    for i, mpos in enumerate(CLOCKWISE_MARKERS):
        if pos == mpos:
            return i
    return -1

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for OrchardLoop:
    - Reward for visiting markers in clockwise order
    - Bonus for completing a full loop
    - Small step penalty for efficiency
    """
    step_penalty = -0.01

    # Get agent positions
    prev_pos = _get_agent_pos(prev_obs)
    next_pos = _get_agent_pos(next_obs)

    # Check if agent moved to a marker
    next_marker_idx = _is_marker_pos(next_pos)
    prev_marker_idx = _is_marker_pos(prev_pos)

    checkpoint_reward = 0.0
    loop_bonus = 0.0

    if next_marker_idx >= 0:
        # Agent is on a marker tile
        # Only reward if this is a NEW marker visit (not staying on same marker)
        if next_pos != prev_pos:
            if _state["last_marker_idx"] is None:
                # First marker visit - reward any marker to get started
                checkpoint_reward = 1.0
                _state["last_marker_idx"] = next_marker_idx
            else:
                # Check if this is the next clockwise marker
                expected_next = (_state["last_marker_idx"] + 1) % len(CLOCKWISE_MARKERS)
                if next_marker_idx == expected_next:
                    checkpoint_reward = 1.0
                    # Check if completed a full loop (returned to first marker)
                    if next_marker_idx == 0 and _state["last_marker_idx"] == len(CLOCKWISE_MARKERS) - 1:
                        loop_bonus = 2.0
                    _state["last_marker_idx"] = next_marker_idx
                elif next_marker_idx == _state["last_marker_idx"]:
                    # Revisiting same marker - no reward, no penalty
                    pass
                else:
                    # Wrong order - small penalty
                    checkpoint_reward = -0.5

    # Noop penalty (discourage standing still)
    noop_penalty = -0.05 if action == 0 else 0.0

    total = checkpoint_reward + loop_bonus + step_penalty + noop_penalty

    return {
        "total": total,
        "checkpoint_reward": checkpoint_reward,
        "loop_bonus": loop_bonus,
        "step_penalty": step_penalty,
        "noop_penalty": noop_penalty,
    }