import numpy as np

# Clockwise checkpoint order based on grid layout:
# #####
# #P m#   -> m is at (row=1, col=3)
# #n#o#   -> n is at (row=2, col=1), o is at (row=2, col=3)
# # p #   -> p is at (row=3, col=2)
# #####
# Clockwise: m(1,3) -> o(2,3) -> p(3,2) -> n(2,1) -> m(1,3) -> ...

CHECKPOINTS_CLOCKWISE = [
    (1, 3),  # m
    (2, 3),  # o
    (3, 2),  # p
    (2, 1),  # n
]

# Per-instance state tracked via a closure dictionary keyed by a hashable obs snapshot
# Since stable-baselines3 calls reward_fn statelessly, we use a global dict
_state = {
    "last_checkpoint_idx": -1,   # index into CHECKPOINTS_CLOCKWISE of last visited checkpoint
    "last_agent_pos": None,       # (row, col) of agent last step
    "steps_since_checkpoint": 0,  # steps since last checkpoint hit
    "total_checkpoints_hit": 0,   # total checkpoints collected (for loop counting)
    "visited_in_loop": set(),     # which checkpoint indices visited this loop
    "prev_obs_id": None,          # to detect env resets
}

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    global _state

    # --- Extract agent position from next_obs ---
    board = next_obs[0]  # shape (5, 5)
    prev_board = prev_obs[0]

    agent_positions = np.argwhere(board == 2.0)
    prev_agent_positions = np.argwhere(prev_board == 2.0)

    if len(agent_positions) == 0:
        return {"total": 0.0, "checkpoint_reward": 0.0, "loop_bonus": 0.0,
                "step_penalty": 0.0, "clockwise_reward": 0.0, "wrong_order_penalty": 0.0}

    agent_pos = tuple(agent_positions[0])  # (row, col)

    if len(prev_agent_positions) > 0:
        prev_pos = tuple(prev_agent_positions[0])
    else:
        prev_pos = agent_pos

    # --- Detect environment reset via prev_obs agent position vs stored ---
    # If agent teleported far (reset), clear state
    if _state["last_agent_pos"] is not None:
        dr = abs(agent_pos[0] - _state["last_agent_pos"][0])
        dc = abs(agent_pos[1] - _state["last_agent_pos"][1])
        if dr + dc > 2:
            # Likely a reset
            _state["last_checkpoint_idx"] = -1
            _state["last_agent_pos"] = None
            _state["steps_since_checkpoint"] = 0
            _state["total_checkpoints_hit"] = 0
            _state["visited_in_loop"] = set()

    _state["last_agent_pos"] = agent_pos
    _state["steps_since_checkpoint"] += 1

    # --- Component rewards ---
    checkpoint_reward = 0.0
    clockwise_reward = 0.0
    wrong_order_penalty = 0.0
    loop_bonus = 0.0
    step_penalty = -0.02  # small penalty each step to encourage efficiency

    # NOOP penalty (action 0)
    noop_penalty = 0.0
    if action == 0:
        noop_penalty = -0.1

    # --- Check if agent is on a checkpoint ---
    if agent_pos in CHECKPOINTS_CLOCKWISE:
        cp_idx = CHECKPOINTS_CLOCKWISE.index(agent_pos)

        # Only reward if this is a new visit (agent just stepped onto it)
        # Check if prev position was NOT this checkpoint (to avoid repeated reward)
        just_arrived = (prev_pos != agent_pos)

        if just_arrived:
            expected_next_idx = (_state["last_checkpoint_idx"] + 1) % len(CHECKPOINTS_CLOCKWISE)

            if cp_idx == expected_next_idx:
                # Correct clockwise order!
                clockwise_reward = 1.5
                checkpoint_reward = 1.0
                _state["last_checkpoint_idx"] = cp_idx
                _state["steps_since_checkpoint"] = 0
                _state["total_checkpoints_hit"] += 1
                _state["visited_in_loop"].add(cp_idx)

                # Bonus for completing a full loop (all 4 checkpoints in order)
                if len(_state["visited_in_loop"]) == 4:
                    loop_bonus = 5.0
                    _state["visited_in_loop"] = set()

            elif _state["last_checkpoint_idx"] == -1:
                # First checkpoint ever — accept any starting point
                checkpoint_reward = 0.5
                _state["last_checkpoint_idx"] = cp_idx
                _state["steps_since_checkpoint"] = 0
                _state["total_checkpoints_hit"] += 1
                _state["visited_in_loop"].add(cp_idx)

            else:
                # Wrong order — penalize backtracking / skipping
                wrong_order_penalty = -1.0

    # --- Movement reward: reward moving toward next checkpoint ---
    progress_reward = 0.0
    if _state["last_checkpoint_idx"] >= 0:
        next_cp_idx = (_state["last_checkpoint_idx"] + 1) % len(CHECKPOINTS_CLOCKWISE)
        next_cp = CHECKPOINTS_CLOCKWISE[next_cp_idx]

        # Manhattan distance to next checkpoint
        prev_dist = abs(prev_pos[0] - next_cp[0]) + abs(prev_pos[1] - next_cp[1])
        curr_dist = abs(agent_pos[0] - next_cp[0]) + abs(agent_pos[1] - next_cp[1])

        if curr_dist < prev_dist:
            progress_reward = 0.1   # moved closer to next checkpoint
        elif curr_dist > prev_dist:
            progress_reward = -0.05  # moved away from next checkpoint

    # --- Total ---
    total = (
        checkpoint_reward
        + clockwise_reward
        + loop_bonus
        + step_penalty
        + noop_penalty
        + wrong_order_penalty
        + progress_reward
    )

    return {
        "total": float(total),
        "checkpoint_reward": float(checkpoint_reward),
        "clockwise_reward": float(clockwise_reward),
        "loop_bonus": float(loop_bonus),
        "step_penalty": float(step_penalty),
        "noop_penalty": float(noop_penalty),
        "wrong_order_penalty": float(wrong_order_penalty),
        "progress_reward": float(progress_reward),
    }