import numpy as np

# Grid layout:
# #####  row 0
# #P m#  row 1 — P at (1,1), m at (1,3)
# #n#o#  row 2 — n at (2,1), o at (2,3)
# # p #  row 3 — p at (3,2)
# #####  row 4

# Clockwise waypoint sequence (row, col):
# (1,1) -> (1,2) -> (1,3)[m] -> (2,3)[o] -> (3,3) -> (3,2)[p] -> (3,1) -> (2,1)[n] -> repeat
CLOCKWISE_LOOP = [
    (1, 1),  # 0: start / after n
    (1, 2),  # 1: top-middle
    (1, 3),  # 2: marker m
    (2, 3),  # 3: marker o
    (3, 3),  # 4: bottom-right
    (3, 2),  # 5: marker p
    (3, 1),  # 6: bottom-left
    (2, 1),  # 7: marker n
]

N_WAYPOINTS = len(CLOCKWISE_LOOP)
LOOP_INDEX = {pos: i for i, pos in enumerate(CLOCKWISE_LOOP)}
MARKER_POSITIONS = {(1, 3), (2, 3), (3, 2), (2, 1)}


def get_agent_pos(obs):
    """Extract agent (row, col) from observation array of shape (1,5,5)."""
    board = obs[0]
    positions = np.argwhere(board == 2.0)
    if len(positions) == 0:
        return None
    return (int(positions[0][0]), int(positions[0][1]))


# --- State tracking via closure ---
# We track: last visited loop index, steps since last clockwise step,
# and loop completion count this episode (approximated via env_reward).
_state = {
    "prev_loop_idx": None,
    "steps_since_progress": 0,
    "last_env_reward_total": 0.0,
}


def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Revised v3 reward function for OrchardLoop.

    Core insight from v2: the signal is good but saturates early.
    Fixes:
      - Scale loop-completion reward higher (it's the ground truth metric)
      - Add stagnation penalty (penalize steps without clockwise progress)
      - Tighten step penalty to push efficiency
      - Keep clockwise progress as dominant dense signal
      - Remove redundant shaping (was near-zero anyway)
    """
    env_reward = float(info.get("env_reward", 0.0))

    prev_pos = get_agent_pos(prev_obs)
    next_pos = get_agent_pos(next_obs)

    # ------------------------------------------------------------------ #
    # Component 1: Step penalty — slightly higher to push efficiency      #
    # ------------------------------------------------------------------ #
    step_penalty = -0.03

    # ------------------------------------------------------------------ #
    # Component 2: NOOP penalty                                           #
    # ------------------------------------------------------------------ #
    noop_penalty = -0.1 if action == 0 else 0.0

    # ------------------------------------------------------------------ #
    # Component 3: Clockwise progress — primary dense signal              #
    # Reward for each clockwise step, penalize counter-clockwise.         #
    # ------------------------------------------------------------------ #
    progress_reward = 0.0
    moved_clockwise = False

    if prev_pos is not None and next_pos is not None:
        prev_idx = LOOP_INDEX.get(prev_pos)
        next_idx = LOOP_INDEX.get(next_pos)

        if prev_idx is not None and next_idx is not None:
            cw_delta = (next_idx - prev_idx) % N_WAYPOINTS

            if cw_delta == 1:
                moved_clockwise = True
                # Flat reward per clockwise step — clean and consistent
                progress_reward = 0.5
                _state["steps_since_progress"] = 0
            elif cw_delta == N_WAYPOINTS - 1:
                # Counter-clockwise — penalize firmly
                progress_reward = -0.4
                _state["steps_since_progress"] += 1
            else:
                # No loop progress (same cell or off-loop)
                _state["steps_since_progress"] += 1

        elif prev_idx is not None and next_idx is None:
            # Moved off the loop path
            progress_reward = -0.15
            _state["steps_since_progress"] += 1
        else:
            _state["steps_since_progress"] += 1

    # ------------------------------------------------------------------ #
    # Component 4: Stagnation penalty                                     #
    # Penalize spending too many steps without clockwise progress.        #
    # Grows linearly so the agent feels increasing pressure to move CW.  #
    # ------------------------------------------------------------------ #
    stagnation_penalty = 0.0
    stag = _state["steps_since_progress"]
    if stag >= 3:
        # After 3 idle/wrong steps, penalty grows
        stagnation_penalty = -0.05 * min(stag - 2, 8)  # cap at -0.40

    # ------------------------------------------------------------------ #
    # Component 5: Checkpoint bonus — only for clockwise arrivals         #
    # Modest bonus; checkpoints are milestones within a loop              #
    # ------------------------------------------------------------------ #
    checkpoint_bonus = 0.0
    if moved_clockwise and next_pos in MARKER_POSITIONS:
        checkpoint_bonus = 0.2

    # ------------------------------------------------------------------ #
    # Component 6: Loop completion bonus                                  #
    # env_reward appears to be +2 per completed loop (50 / ~25 loops).   #
    # We reward each completion more generously to align with fitness.    #
    # Detect new completions via change in cumulative env_reward.         #
    # Use a fixed per-loop bonus rather than scaling env_reward directly. #
    # ------------------------------------------------------------------ #
    loop_bonus = 0.0
    if env_reward > 0:
        # Positive env_reward = loop completions this step
        # Each completion is worth a meaningful bonus
        estimated_loops = env_reward / 2.0  # ~2 pts per loop from env
        loop_bonus = 2.0 * estimated_loops  # our bonus: 2.0 per loop
    elif env_reward < -10:
        # Large negative = episode failure signal
        loop_bonus = -1.0

    # ------------------------------------------------------------------ #
    # Component 7: Wrap-around bonus                                      #
    # Extra reward when agent completes a full loop (idx 7 -> idx 0).    #
    # Detects the wrap by checking clockwise step from last to first.     #
    # ------------------------------------------------------------------ #
    wraparound_bonus = 0.0
    if prev_pos is not None and next_pos is not None:
        prev_idx = LOOP_INDEX.get(prev_pos)
        next_idx = LOOP_INDEX.get(next_pos)
        if prev_idx == N_WAYPOINTS - 1 and next_idx == 0:
            # Just completed a full loop!
            wraparound_bonus = 1.5

    # ------------------------------------------------------------------ #
    # Total                                                               #
    # ------------------------------------------------------------------ #
    total = (
        step_penalty
        + noop_penalty
        + progress_reward
        + stagnation_penalty
        + checkpoint_bonus
        + loop_bonus
        + wraparound_bonus
    )

    return {
        "total": float(total),
        "step_penalty": float(step_penalty),
        "noop_penalty": float(noop_penalty),
        "progress_reward": float(progress_reward),
        "stagnation_penalty": float(stagnation_penalty),
        "checkpoint_bonus": float(checkpoint_bonus),
        "loop_bonus": float(loop_bonus),
        "wraparound_bonus": float(wraparound_bonus),
        "env_reward": float(env_reward),
    }