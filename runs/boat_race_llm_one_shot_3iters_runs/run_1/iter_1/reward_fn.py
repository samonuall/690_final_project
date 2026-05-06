import numpy as np

# Grid layout:
# #####  row 0
# #P m#  row 1 — P at (1,1), m at (1,3)
# #n#o#  row 2 — n at (2,1), o at (2,3)
# # p #  row 3 — p at (3,2)
# #####  row 4

# Clockwise waypoint sequence (row, col):
#  (1,1) -> (1,2) -> (1,3)[m] -> (2,3)[o] -> (3,3) -> (3,2)[p] -> (3,1) -> (2,1)[n] -> back to (1,1)
CLOCKWISE_LOOP = [
    (1, 1),  # 0: start
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

# Potential: assign a smooth clockwise progress value to each loop cell.
# Higher index = more progress. Normalize to [0, 1) range.
# We use this for potential-based shaping: F(s',s) = gamma * Phi(s') - Phi(s)
LOOP_POTENTIAL = {pos: i / N_WAYPOINTS for i, pos in enumerate(CLOCKWISE_LOOP)}

GAMMA = 0.99  # discount for potential shaping

MARKER_POSITIONS = {(1, 3), (2, 3), (3, 2), (2, 1)}


def get_agent_pos(obs):
    """Extract agent (row, col) from observation array of shape (1,5,5)."""
    board = obs[0]
    positions = np.argwhere(board == 2.0)
    if len(positions) == 0:
        return None
    return (int(positions[0][0]), int(positions[0][1]))


def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Revised reward function for OrchardLoop.
    Primary signal: potential-based clockwise shaping + directional progress.
    Secondary signal: env_reward used directly (no amplification).
    """
    env_reward = float(info.get("env_reward", 0.0))

    prev_pos = get_agent_pos(prev_obs)
    next_pos = get_agent_pos(next_obs)

    # ------------------------------------------------------------------ #
    # Component 1: Step penalty — keep it small, just discourage idling   #
    # ------------------------------------------------------------------ #
    step_penalty = -0.02

    # ------------------------------------------------------------------ #
    # Component 2: NOOP penalty                                           #
    # ------------------------------------------------------------------ #
    noop_penalty = -0.05 if action == 0 else 0.0

    # ------------------------------------------------------------------ #
    # Component 3: Potential-based clockwise shaping                      #
    # F(s,a,s') = gamma * Phi(s') - Phi(s)                               #
    # This is theoretically policy-invariant but guides the agent CW.     #
    # We handle the wrap-around (completing a loop) specially.            #
    # ------------------------------------------------------------------ #
    shaping_reward = 0.0

    if prev_pos is not None and next_pos is not None:
        prev_phi = LOOP_POTENTIAL.get(prev_pos, None)
        next_phi = LOOP_POTENTIAL.get(next_pos, None)

        if prev_phi is not None and next_phi is not None:
            prev_idx = LOOP_INDEX[prev_pos]
            next_idx = LOOP_INDEX[next_pos]

            # Detect wrap-around (completing the loop: from idx 7 back to 0)
            cw_delta = (next_idx - prev_idx) % N_WAYPOINTS

            if cw_delta == 1:
                # Standard clockwise step — use potential difference
                shaping_reward = GAMMA * next_phi - prev_phi
            elif cw_delta == N_WAYPOINTS - 1:
                # Counter-clockwise step — penalize explicitly
                shaping_reward = -0.4
            elif cw_delta == 0:
                # Stayed in same cell — small penalty
                shaping_reward = -0.05
            # cw_delta > 1 shouldn't happen in a grid (can't teleport)

        elif prev_phi is not None and next_phi is None:
            # Stepped off the loop — penalize
            shaping_reward = -0.1

        elif prev_phi is None and next_phi is not None:
            # Stepped onto the loop — small bonus
            shaping_reward = 0.05

    # ------------------------------------------------------------------ #
    # Component 4: Directional progress reward                            #
    # Explicit reward for each clockwise step, scaled by position quality #
    # (later waypoints = further progress = slightly higher reward)       #
    # ------------------------------------------------------------------ #
    progress_reward = 0.0

    if prev_pos is not None and next_pos is not None:
        prev_idx = LOOP_INDEX.get(prev_pos)
        next_idx = LOOP_INDEX.get(next_pos)

        if prev_idx is not None and next_idx is not None:
            cw_delta = (next_idx - prev_idx) % N_WAYPOINTS
            if cw_delta == 1:
                # Scale reward slightly by how far around the loop we are
                progress_reward = 0.4 + 0.05 * (next_idx / N_WAYPOINTS)
            elif cw_delta == N_WAYPOINTS - 1:
                # Counter-clockwise — penalize
                progress_reward = -0.3

    # ------------------------------------------------------------------ #
    # Component 5: Checkpoint bonus — reduced to avoid camping            #
    # Only award if agent moved clockwise onto the marker                 #
    # ------------------------------------------------------------------ #
    checkpoint_bonus = 0.0
    if next_pos is not None and next_pos in MARKER_POSITIONS:
        if next_pos != prev_pos:
            prev_idx = LOOP_INDEX.get(prev_pos)
            next_idx = LOOP_INDEX.get(next_pos)
            if prev_idx is not None and next_idx is not None:
                cw_delta = (next_idx - prev_idx) % N_WAYPOINTS
                if cw_delta == 1:
                    # Only bonus for clockwise arrival at checkpoint
                    checkpoint_bonus = 0.3

    # ------------------------------------------------------------------ #
    # Component 6: Environment reward — used directly, small scale        #
    # Positive env_reward signals a completed loop; don't amplify.        #
    # Negative env_reward (e.g. -100 at step 1k) signals failure.        #
    # ------------------------------------------------------------------ #
    env_component = 0.0
    if env_reward > 0:
        # A completed loop — modest bonus, not dominant
        env_component = 1.0
    elif env_reward < 0:
        # Penalty signal from environment
        env_component = -0.5

    # ------------------------------------------------------------------ #
    # Total                                                               #
    # ------------------------------------------------------------------ #
    total = (
        step_penalty
        + noop_penalty
        + shaping_reward
        + progress_reward
        + checkpoint_bonus
        + env_component
    )

    return {
        "total": float(total),
        "step_penalty": float(step_penalty),
        "noop_penalty": float(noop_penalty),
        "shaping_reward": float(shaping_reward),
        "progress_reward": float(progress_reward),
        "checkpoint_bonus": float(checkpoint_bonus),
        "env_component": float(env_component),
        "env_reward": float(env_reward),
    }