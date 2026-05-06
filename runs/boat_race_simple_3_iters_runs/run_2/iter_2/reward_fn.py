import numpy as np

# Clockwise marker positions (row, col) in the 5x5 grid:
# #####
# #P m#   -> m at (1,3)
# #n#o#   -> n at (2,1), o at (2,3)
# # p #   -> p at (3,2)
# #####
# Clockwise order: m(1,3) -> o(2,3) -> p(3,2) -> n(2,1) -> back to m
CLOCKWISE_MARKERS = [
    (1, 3),  # m  index 0
    (2, 3),  # o  index 1
    (3, 2),  # p  index 2
    (2, 1),  # n  index 3
]
N_MARKERS = len(CLOCKWISE_MARKERS)

# Per-instance state keyed by object id of obs array to handle
# multiple envs — use a simple single-state approach with episode
# detection via agent teleport back to start.
_state = {
    "next_expected": 0,    # which marker index we expect next
    "markers_hit": 0,      # how many markers hit in current loop
    "prev_agent_pos": None,
}

def _get_agent_pos(obs):
    board = obs[0]
    locs = np.argwhere(board == 2.0)
    if len(locs) > 0:
        return (int(locs[0][0]), int(locs[0][1]))
    return None

def _marker_idx(pos):
    if pos is None:
        return -1
    for i, mpos in enumerate(CLOCKWISE_MARKERS):
        if pos == mpos:
            return i
    return -1

def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    prev_pos = _get_agent_pos(prev_obs)
    next_pos = _get_agent_pos(next_obs)

    # --- Episode reset detection ---
    # If agent position jumped by more than 1 cell, a reset occurred
    if (_state["prev_agent_pos"] is not None
            and prev_pos is not None
            and _manhattan(_state["prev_agent_pos"], prev_pos) > 1):
        _state["next_expected"] = 0
        _state["markers_hit"] = 0

    _state["prev_agent_pos"] = next_pos

    # --- Core logic ---
    checkpoint_reward = 0.0
    loop_bonus = 0.0

    # Only process if agent actually moved
    if next_pos is not None and next_pos != prev_pos:
        idx = _marker_idx(next_pos)
        if idx >= 0:
            if idx == _state["next_expected"]:
                # Correct next clockwise marker
                _state["markers_hit"] += 1
                _state["next_expected"] = (idx + 1) % N_MARKERS

                if _state["markers_hit"] % N_MARKERS == 0:
                    # Completed a full loop!
                    loop_bonus = 5.0
                else:
                    # Intermediate checkpoint reward (smaller)
                    checkpoint_reward = 0.5
            else:
                # Wrong marker — reset progress, must start over
                _state["next_expected"] = 0
                _state["markers_hit"] = 0
                # If they landed on marker 0, let them start fresh from here
                if idx == 0:
                    _state["next_expected"] = 1
                    _state["markers_hit"] = 1
                    checkpoint_reward = 0.5

    # Step penalty — keep it tiny so checkpoints are still clearly positive
    step_penalty = -0.02

    total = checkpoint_reward + loop_bonus + step_penalty

    return {
        "total": total,
        "checkpoint_reward": checkpoint_reward,
        "loop_bonus": loop_bonus,
        "step_penalty": step_penalty,
    }