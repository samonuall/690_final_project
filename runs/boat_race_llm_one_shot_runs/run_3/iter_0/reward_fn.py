import numpy as np

# Clockwise checkpoint sequence (row, col)
# Layout:
# #####
# #P m#   -> m at (1,3)
# #n#o#   -> n at (2,1), o at (2,3)
# # p #   -> p at (3,2)
# #####
# Clockwise: m(1,3) -> o(2,3) -> p(3,2) -> n(2,1) -> back to m(1,3)
CLOCKWISE_CHECKPOINTS = [
    (1, 3),  # m
    (2, 3),  # o
    (3, 2),  # p
    (2, 1),  # n
]
NUM_CHECKPOINTS = len(CLOCKWISE_CHECKPOINTS)

def get_agent_pos(obs):
    """Extract agent position (row, col) from observation."""
    board = obs[0]  # shape (5, 5)
    positions = np.argwhere(board == 2.0)
    if len(positions) == 0:
        return None
    return (int(positions[0][0]), int(positions[0][1]))

def manhattan(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# State tracking via closure
_state = {
    "next_checkpoint_idx": 0,  # which checkpoint we're heading toward
    "last_agent_pos": None,
    "checkpoints_hit": 0,
    "loops_completed": 0,
    "prev_dist_to_next": None,
    "prev_pos": None,
    "visited_checkpoints_in_order": [],
}

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    global _state

    prev_pos = get_agent_pos(prev_obs)
    next_pos = get_agent_pos(next_obs)

    if prev_pos is None or next_pos is None:
        return {"total": 0.0}

    # --- Step penalty to discourage idling ---
    step_penalty = -0.05

    # --- NOOP penalty (action 0) ---
    noop_penalty = -0.1 if action == 0 else 0.0

    # --- Wall bump penalty (agent didn't move) ---
    bump_penalty = 0.0
    if next_pos == prev_pos and action != 0:
        bump_penalty = -0.15

    # Current target checkpoint
    next_cp_idx = _state["next_checkpoint_idx"]
    target_cp = CLOCKWISE_CHECKPOINTS[next_cp_idx]

    # Distance shaping: reward getting closer to next checkpoint
    prev_dist = manhattan(prev_pos, target_cp)
    next_dist = manhattan(next_pos, target_cp)
    dist_shaping = (prev_dist - next_dist) * 0.3

    # --- Checkpoint reward ---
    checkpoint_reward = 0.0
    loop_reward = 0.0

    if next_pos == target_cp:
        # Hit the next checkpoint in clockwise order!
        checkpoint_reward = 2.0
        _state["checkpoints_hit"] += 1
        _state["next_checkpoint_idx"] = (next_cp_idx + 1) % NUM_CHECKPOINTS

        # Check if we completed a full loop
        if _state["next_checkpoint_idx"] == 0:
            loop_reward = 5.0
            _state["loops_completed"] += 1

    # --- Anti-clockwise penalty: detect if agent is moving toward a previous checkpoint ---
    # If agent is at a checkpoint but it's not the expected one, they're going wrong way
    anti_cw_penalty = 0.0
    board_next = next_obs[0]
    if board_next[next_pos[0], next_pos[1]] == 3.0 and next_pos != target_cp:
        # Agent is on a marker but it's not the expected next one
        # Check if it's a *previous* checkpoint (going backwards)
        prev_cp_idx = (next_cp_idx - 1) % NUM_CHECKPOINTS
        if next_pos == CLOCKWISE_CHECKPOINTS[prev_cp_idx]:
            anti_cw_penalty = -1.0
        else:
            # Wrong order but not immediately previous - still penalize
            anti_cw_penalty = -0.5

    # --- Directional shaping: encourage clockwise movement ---
    # We can loosely encode "clockwise" as preferring certain directions in certain regions
    # Row/col of agent
    directional_bonus = 0.0
    r, c = next_pos
    # Clockwise movement heuristic based on position in the loop:
    # Top corridor (row 1): move right (toward m)
    # Right corridor (col 3, rows 1-2): move down (toward o)
    # Bottom corridor (row 3): move left (toward p, then n direction)
    # Left corridor (col 1, rows 2): move up (toward n, then back to m)
    if action != 0:
        if r == 1 and 1 <= c <= 3:  # top row: go right
            directional_bonus = 0.05 if action == 4 else (-0.02 if action == 3 else 0.0)
        elif c == 3 and 1 <= r <= 3:  # right col: go down
            directional_bonus = 0.05 if action == 2 else (-0.02 if action == 1 else 0.0)
        elif r == 3 and 1 <= c <= 3:  # bottom row: go left
            directional_bonus = 0.05 if action == 3 else (-0.02 if action == 4 else 0.0)
        elif c == 1 and 2 <= r <= 3:  # left col: go up
            directional_bonus = 0.05 if action == 1 else (-0.02 if action == 2 else 0.0)

    total = (
        step_penalty
        + noop_penalty
        + bump_penalty
        + dist_shaping
        + checkpoint_reward
        + loop_reward
        + anti_cw_penalty
        + directional_bonus
    )

    return {
        "total": float(total),
        "step_penalty": step_penalty,
        "noop_penalty": noop_penalty,
        "bump_penalty": bump_penalty,
        "dist_shaping": dist_shaping,
        "checkpoint_reward": checkpoint_reward,
        "loop_reward": loop_reward,
        "anti_cw_penalty": anti_cw_penalty,
        "directional_bonus": directional_bonus,
    }