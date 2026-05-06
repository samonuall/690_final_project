import numpy as np

# Grid layout:
# #####
# #P m#   -> m is at (row=1, col=3)
# #n#o#   -> n is at (row=2, col=1), o is at (row=2, col=3)
# # p #   -> p is at (row=3, col=2)
# #####
# Clockwise order: m(1,3) -> o(2,3) -> p(3,2) -> n(2,1) -> back to m

CHECKPOINTS_CLOCKWISE = [
    (1, 3),  # m
    (2, 3),  # o
    (3, 2),  # p
    (2, 1),  # n
]

_state = {
    "last_checkpoint_idx": -1,
    "last_agent_pos":       None,
    "visited_in_loop":      set(),
    "loops_completed":      0,
    "initialized":          False,
}


def _reset_state():
    _state["last_checkpoint_idx"] = -1
    _state["last_agent_pos"]      = None
    _state["visited_in_loop"]     = set()
    _state["loops_completed"]     = 0
    _state["initialized"]         = True


def reward_fn(prev_obs, action, next_obs, info) -> dict:
    global _state

    # ------------------------------------------------------------------ #
    # 0. Extract agent positions
    # ------------------------------------------------------------------ #
    board      = next_obs[0]   # (5, 5)
    prev_board = prev_obs[0]

    agent_pos_arr = np.argwhere(board == 2.0)
    prev_pos_arr  = np.argwhere(prev_board == 2.0)

    null_return = {
        "total": 0.0, "checkpoint_reward": 0.0,
        "loop_bonus": 0.0, "step_penalty": 0.0,
        "noop_penalty": 0.0, "wrong_order_penalty": 0.0,
        "progress_reward": 0.0,
    }

    if len(agent_pos_arr) == 0:
        return null_return

    agent_pos = tuple(agent_pos_arr[0])
    prev_pos  = tuple(prev_pos_arr[0]) if len(prev_pos_arr) > 0 else agent_pos

    # ------------------------------------------------------------------ #
    # 1. Detect reset
    # ------------------------------------------------------------------ #
    if not _state["initialized"]:
        _reset_state()

    if _state["last_agent_pos"] is not None:
        dr = abs(agent_pos[0] - _state["last_agent_pos"][0])
        dc = abs(agent_pos[1] - _state["last_agent_pos"][1])
        if dr + dc > 2:
            _reset_state()

    _state["last_agent_pos"] = agent_pos

    # ------------------------------------------------------------------ #
    # 2. Step penalty — small, fixed, encourages speed without dominating
    # ------------------------------------------------------------------ #
    step_penalty = -0.02

    # ------------------------------------------------------------------ #
    # 3. NOOP penalty — firm but not extreme
    # ------------------------------------------------------------------ #
    noop_penalty = -0.15 if action == 0 else 0.0

    # ------------------------------------------------------------------ #
    # 4. Checkpoint reward — only on arrival, only correct clockwise order
    # ------------------------------------------------------------------ #
    checkpoint_reward   = 0.0
    wrong_order_penalty = 0.0
    loop_bonus          = 0.0

    just_arrived = (prev_pos != agent_pos) and (agent_pos in CHECKPOINTS_CLOCKWISE)

    if just_arrived:
        cp_idx = CHECKPOINTS_CLOCKWISE.index(agent_pos)

        if _state["last_checkpoint_idx"] == -1:
            # Accept any first checkpoint to bootstrap learning
            checkpoint_reward = 0.5
            _state["last_checkpoint_idx"] = cp_idx
            _state["visited_in_loop"].add(cp_idx)

        else:
            expected = (_state["last_checkpoint_idx"] + 1) % len(CHECKPOINTS_CLOCKWISE)

            if cp_idx == expected:
                # Correct clockwise step
                checkpoint_reward = 1.0
                _state["last_checkpoint_idx"] = cp_idx
                _state["visited_in_loop"].add(cp_idx)

                # Full loop completed
                if len(_state["visited_in_loop"]) == 4:
                    _state["loops_completed"] += 1
                    # Flat loop bonus — no escalation, no exploitation
                    loop_bonus = 8.0
                    _state["visited_in_loop"] = set()

            else:
                # Wrong direction / backtrack
                wrong_order_penalty = -1.5

    # ------------------------------------------------------------------ #
    # 5. Dense progress shaping — small potential-based signal
    #    Guides agent toward the NEXT checkpoint without overwhelming
    #    the sparse checkpoint/loop rewards
    # ------------------------------------------------------------------ #
    progress_reward = 0.0
    if _state["last_checkpoint_idx"] >= 0:
        next_cp_idx = (_state["last_checkpoint_idx"] + 1) % len(CHECKPOINTS_CLOCKWISE)
        next_cp     = CHECKPOINTS_CLOCKWISE[next_cp_idx]

        prev_dist = abs(prev_pos[0] - next_cp[0]) + abs(prev_pos[1] - next_cp[1])
        curr_dist = abs(agent_pos[0] - next_cp[0]) + abs(agent_pos[1] - next_cp[1])

        # Small, symmetric shaping: reward closing distance, penalise opening it
        if curr_dist < prev_dist:
            progress_reward = 0.1
        elif curr_dist > prev_dist:
            progress_reward = -0.05

    # ------------------------------------------------------------------ #
    # 6. Anchor to environment reward
    #    env_reward fires on loop completions in the ground-truth sense;
    #    adding it directly keeps our reward correlated with fitness
    # ------------------------------------------------------------------ #
    env_reward = float(info.get("env_reward", 0.0))

    # ------------------------------------------------------------------ #
    # 7. Assemble total
    # ------------------------------------------------------------------ #
    total = (
        checkpoint_reward
        + loop_bonus
        + step_penalty
        + noop_penalty
        + wrong_order_penalty
        + progress_reward
        + env_reward          # ground-truth anchor
    )

    return {
        "total":               float(total),
        "checkpoint_reward":   float(checkpoint_reward),
        "loop_bonus":          float(loop_bonus),
        "step_penalty":        float(step_penalty),
        "noop_penalty":        float(noop_penalty),
        "wrong_order_penalty": float(wrong_order_penalty),
        "progress_reward":     float(progress_reward),
        "env_reward":          float(env_reward),
    }