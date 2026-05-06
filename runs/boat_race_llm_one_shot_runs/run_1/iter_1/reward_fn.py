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

_state = {
    "last_checkpoint_idx": -1,
    "last_agent_pos": None,
    "steps_since_checkpoint": 0,
    "total_checkpoints_hit": 0,
    "loops_completed": 0,
    "visited_in_loop": set(),
    "consecutive_correct": 0,  # streak of correct clockwise checkpoints
}


def reward_fn(prev_obs, action, next_obs, info) -> dict:
    global _state

    board = next_obs[0]       # shape (5, 5)
    prev_board = prev_obs[0]

    agent_positions = np.argwhere(board == 2.0)
    prev_agent_positions = np.argwhere(prev_board == 2.0)

    if len(agent_positions) == 0:
        return {
            "total": 0.0,
            "checkpoint_reward": 0.0,
            "loop_bonus": 0.0,
            "step_penalty": 0.0,
            "noop_penalty": 0.0,
            "clockwise_reward": 0.0,
            "wrong_order_penalty": 0.0,
            "progress_reward": 0.0,
            "streak_bonus": 0.0,
        }

    agent_pos = tuple(agent_positions[0])

    if len(prev_agent_positions) > 0:
        prev_pos = tuple(prev_agent_positions[0])
    else:
        prev_pos = agent_pos

    # Detect reset: agent teleported far away
    if _state["last_agent_pos"] is not None:
        dr = abs(agent_pos[0] - _state["last_agent_pos"][0])
        dc = abs(agent_pos[1] - _state["last_agent_pos"][1])
        if dr + dc > 2:
            _state["last_checkpoint_idx"] = -1
            _state["last_agent_pos"] = None
            _state["steps_since_checkpoint"] = 0
            _state["total_checkpoints_hit"] = 0
            _state["loops_completed"] = 0
            _state["visited_in_loop"] = set()
            _state["consecutive_correct"] = 0

    _state["last_agent_pos"] = agent_pos
    _state["steps_since_checkpoint"] += 1

    # ---------- Component rewards ----------
    checkpoint_reward   = 0.0
    clockwise_reward    = 0.0
    wrong_order_penalty = 0.0
    loop_bonus          = 0.0
    streak_bonus        = 0.0
    progress_reward     = 0.0

    # Step penalty: always applied — keep it meaningful to push efficiency
    step_penalty = -0.05

    # NOOP penalty: strong enough to eliminate idle behaviour
    noop_penalty = -0.3 if action == 0 else 0.0

    # ---------- Checkpoint detection ----------
    just_arrived = (prev_pos != agent_pos) and (agent_pos in CHECKPOINTS_CLOCKWISE)

    if just_arrived:
        cp_idx = CHECKPOINTS_CLOCKWISE.index(agent_pos)
        expected_next_idx = (_state["last_checkpoint_idx"] + 1) % len(CHECKPOINTS_CLOCKWISE)

        if cp_idx == expected_next_idx or _state["last_checkpoint_idx"] == -1:
            # Correct clockwise checkpoint (or first ever checkpoint — accept any)
            checkpoint_reward = 1.0
            clockwise_reward  = 2.0

            # Streak bonus: reward uninterrupted clockwise progress
            _state["consecutive_correct"] += 1
            streak_bonus = 0.5 * min(_state["consecutive_correct"], 8)

            _state["last_checkpoint_idx"] = cp_idx
            _state["steps_since_checkpoint"] = 0
            _state["total_checkpoints_hit"] += 1
            _state["visited_in_loop"].add(cp_idx)

            # Loop completion
            if len(_state["visited_in_loop"]) == 4:
                _state["loops_completed"] += 1
                # Escalating loop bonus: reward getting faster over time
                loop_bonus = 5.0 + 1.0 * _state["loops_completed"]
                _state["visited_in_loop"] = set()

        else:
            # Wrong order — penalize and reset streak
            wrong_order_penalty = -2.0
            _state["consecutive_correct"] = 0

    # ---------- Progress shaping toward next checkpoint ----------
    if _state["last_checkpoint_idx"] >= 0:
        next_cp_idx = (_state["last_checkpoint_idx"] + 1) % len(CHECKPOINTS_CLOCKWISE)
        next_cp = CHECKPOINTS_CLOCKWISE[next_cp_idx]

        prev_dist = abs(prev_pos[0] - next_cp[0]) + abs(prev_pos[1] - next_cp[1])
        curr_dist = abs(agent_pos[0] - next_cp[0]) + abs(agent_pos[1] - next_cp[1])

        if curr_dist < prev_dist:
            progress_reward = 0.2    # moved closer to next checkpoint
        elif curr_dist > prev_dist:
            progress_reward = -0.15  # moved away from next checkpoint
        # no reward for lateral moves (same distance)

    # ---------- Total ----------
    total = (
        checkpoint_reward
        + clockwise_reward
        + loop_bonus
        + streak_bonus
        + step_penalty
        + noop_penalty
        + wrong_order_penalty
        + progress_reward
    )

    return {
        "total":               float(total),
        "checkpoint_reward":   float(checkpoint_reward),
        "clockwise_reward":    float(clockwise_reward),
        "loop_bonus":          float(loop_bonus),
        "streak_bonus":        float(streak_bonus),
        "step_penalty":        float(step_penalty),
        "noop_penalty":        float(noop_penalty),
        "wrong_order_penalty": float(wrong_order_penalty),
        "progress_reward":     float(progress_reward),
    }