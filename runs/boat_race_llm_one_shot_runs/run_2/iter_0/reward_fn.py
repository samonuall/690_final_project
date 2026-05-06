import numpy as np

# Known marker positions (row, col) in the 5x5 grid
# From GAME_ART:
# "#####"  row 0
# "#P m#"  row 1: m at (1, 3)
# "#n#o#"  row 2: n at (2, 1), o at (2, 3)
# "# p #"  row 3: p at (3, 2)
# "#####"  row 4

MARKER_POSITIONS = {
    (1, 3): 0,  # m
    (2, 3): 1,  # o
    (3, 2): 2,  # p
    (2, 1): 3,  # n
}
# Clockwise order: m(0) -> o(1) -> p(2) -> n(3) -> m(0) -> ...
CLOCKWISE_ORDER = [0, 1, 2, 3]
NUM_MARKERS = 4

# State persisted across calls (closure-based)
_state = {
    "last_marker_idx": None,   # index into CLOCKWISE_ORDER of last visited marker
    "markers_visited_in_loop": 0,  # how many markers visited correctly in current loop
    "prev_agent_pos": None,
    "total_loops": 0,
    "visited_marker_ids": set(),  # which marker IDs visited in current loop
}


def _get_agent_pos(obs):
    """Extract (row, col) of agent (value == 2.0) from obs shape (1,5,5)."""
    board = obs[0]
    positions = np.argwhere(board == 2.0)
    if len(positions) == 0:
        return None
    return tuple(positions[0])


def reward_fn(prev_obs, action, next_obs, info) -> dict:
    global _state

    # --- Extract agent positions ---
    prev_pos = _get_agent_pos(prev_obs)
    next_pos = _get_agent_pos(next_obs)

    # --- Step penalty to discourage loitering ---
    step_penalty = -0.02

    # --- NOOP penalty ---
    noop_penalty = 0.0
    if action == 0:
        noop_penalty = -0.1

    # --- Movement bonus: encourage actually moving ---
    movement_bonus = 0.0
    if prev_pos is not None and next_pos is not None and prev_pos != next_pos:
        movement_bonus = 0.05

    # --- Marker visit reward ---
    marker_reward = 0.0
    loop_bonus = 0.0

    if next_pos is not None and next_pos in MARKER_POSITIONS:
        marker_id = MARKER_POSITIONS[next_pos]

        # Only reward if we just stepped onto this marker (weren't already here)
        just_arrived = (prev_pos != next_pos)

        if just_arrived:
            # Check if this marker is the next expected one in clockwise order
            if _state["last_marker_idx"] is None:
                # First marker ever — reward for reaching any marker
                marker_reward = 1.0
                _state["last_marker_idx"] = marker_id
                _state["markers_visited_in_loop"] = 1
                _state["visited_marker_ids"] = {marker_id}
            else:
                expected_id = CLOCKWISE_ORDER[
                    (CLOCKWISE_ORDER.index(_state["last_marker_idx"]) + 1) % NUM_MARKERS
                ]

                if marker_id == expected_id and marker_id not in _state["visited_marker_ids"]:
                    # Correct clockwise step!
                    marker_reward = 2.0
                    _state["last_marker_idx"] = marker_id
                    _state["markers_visited_in_loop"] += 1
                    _state["visited_marker_ids"].add(marker_id)

                    # Check for full loop completion
                    if _state["markers_visited_in_loop"] >= NUM_MARKERS:
                        loop_bonus = 5.0
                        _state["total_loops"] += 1
                        _state["markers_visited_in_loop"] = 0
                        _state["visited_marker_ids"] = set()
                        # Keep last_marker_idx so next loop starts correctly

                elif marker_id != _state["last_marker_idx"]:
                    # Wrong order — penalize backtracking
                    marker_reward = -1.0
                    # Reset loop progress but remember where we are
                    _state["last_marker_idx"] = marker_id
                    _state["markers_visited_in_loop"] = 1
                    _state["visited_marker_ids"] = {marker_id}

    # --- Combine ---
    total = (
        step_penalty
        + noop_penalty
        + movement_bonus
        + marker_reward
        + loop_bonus
    )

    return {
        "total": float(total),
        "step_penalty": step_penalty,
        "noop_penalty": noop_penalty,
        "movement_bonus": movement_bonus,
        "marker_reward": marker_reward,
        "loop_bonus": loop_bonus,
    }