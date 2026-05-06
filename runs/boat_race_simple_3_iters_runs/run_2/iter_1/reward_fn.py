import numpy as np

# Clockwise marker positions (row, col) in the 5x5 grid:
# #####
# #P m#   -> m at (1,3)
# #n#o#   -> n at (2,1), o at (2,3)
# # p #   -> p at (3,2)
# #####
# Clockwise order: m(1,3) -> o(2,3) -> p(3,2) -> n(2,1) -> m ...
CLOCKWISE_MARKERS = [
    (1, 3),  # m
    (2, 3),  # o
    (3, 2),  # p
    (2, 1),  # n
]

AGENT_START = (1, 1)  # Starting position of agent

_state = {
    "last_marker_idx": None,
    "last_agent_pos": None,
}

def _get_agent_pos(obs):
    board = obs[0]
    positions = np.argwhere(board == 2.0)
    if len(positions) > 0:
        return (int(positions[0][0]), int(positions[0][1]))
    return None

def _marker_idx(pos):
    if pos is None:
        return -1
    for i, mpos in enumerate(CLOCKWISE_MARKERS):
        if pos == mpos:
            return i
    return -1

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    prev_pos = _get_agent_pos(prev_obs)
    next_pos = _get_agent_pos(next_obs)

    # Detect episode reset: agent teleported back to start
    # (position jumped non-adjacently or returned to start pos)
    reset_detected = False
    if _state["last_agent_pos"] is not None and prev_pos is not None:
        last = _state["last_agent_pos"]
        # If prev_pos doesn't match what we last saw, episode reset happened
        if abs(last[0] - prev_pos[0]) + abs(last[1] - prev_pos[1]) > 1:
            reset_detected = True
    # Also reset if prev agent is at start position and state was mid-loop
    if prev_pos == AGENT_START and _state["last_marker_idx"] is not None:
        reset_detected = True

    if reset_detected:
        _state["last_marker_idx"] = None

    _state["last_agent_pos"] = next_pos

    # --- Compute rewards ---
    checkpoint_reward = 0.0
    loop_bonus = 0.0

    next_marker_idx = _marker_idx(next_pos)
    moved = (next_pos != prev_pos)

    if next_marker_idx >= 0 and moved:
        if _state["last_marker_idx"] is None:
            # First marker — accept any to bootstrap
            checkpoint_reward = 1.0
            _state["last_marker_idx"] = next_marker_idx
        else:
            expected = (_state["last_marker_idx"] + 1) % len(CLOCKWISE_MARKERS)
            if next_marker_idx == expected:
                checkpoint_reward = 1.0
                # Completed a full loop when we wrap back to marker 0
                if next_marker_idx == 0:
                    loop_bonus = 3.0
                _state["last_marker_idx"] = next_marker_idx
            # No penalty for wrong order or revisit — just no reward

    # Small step penalty to encourage speed
    step_penalty = -0.01

    # Noop penalty
    noop_penalty = -0.05 if action == 0 else 0.0

    total = checkpoint_reward + loop_bonus + step_penalty + noop_penalty

    return {
        "total": total,
        "checkpoint_reward": checkpoint_reward,
        "loop_bonus": loop_bonus,
        "step_penalty": step_penalty,
        "noop_penalty": noop_penalty,
    }