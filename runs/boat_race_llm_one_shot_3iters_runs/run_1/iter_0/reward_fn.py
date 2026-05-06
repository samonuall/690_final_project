import numpy as np

# Known positions of markers in clockwise order (row, col):
# Start area -> m (top-right) -> o (mid-right) -> p (bottom-mid) -> n (mid-left) -> back
# Full clockwise path positions (approximate walkable cells in order):
# (1,1) -> (1,2) -> (1,3) -> (2,3) -> (3,3) -> (3,2) -> (3,1) -> (2,1) -> (1,1)
# Markers: m=(1,3), o=(2,3), p=(3,2), n=(2,1)

# Define the clockwise loop as an ordered sequence of (row, col) waypoints
CLOCKWISE_LOOP = [
    (1, 1),  # start / agent initial pos
    (1, 2),  # top middle
    (1, 3),  # marker m (top-right)
    (2, 3),  # marker o (mid-right)
    (3, 3),  # bottom-right corner
    (3, 2),  # marker p (bottom-mid)
    (3, 1),  # bottom-left corner
    (2, 1),  # marker n (mid-left)
]

N_WAYPOINTS = len(CLOCKWISE_LOOP)

# Map each position to its index in the clockwise loop
LOOP_INDEX = {pos: i for i, pos in enumerate(CLOCKWISE_LOOP)}

# Marker positions (for checkpoint detection)
MARKER_POSITIONS = {(1, 3), (2, 3), (3, 2), (2, 1)}

def get_agent_pos(obs):
    """Extract agent (row, col) from observation array of shape (1,5,5)."""
    board = obs[0]  # shape (5,5)
    positions = np.argwhere(board == 2.0)
    if len(positions) == 0:
        return None
    return (int(positions[0][0]), int(positions[0][1]))


def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for the OrchardLoop environment.
    Encourages clockwise navigation around the loop.
    """
    # Base environment reward (sparse)
    env_reward = info.get("env_reward", 0.0)

    prev_pos = get_agent_pos(prev_obs)
    next_pos = get_agent_pos(next_obs)

    # --- Component 1: Step penalty to discourage idling ---
    step_penalty = -0.01

    # --- Component 2: NOOP penalty (discourage doing nothing) ---
    noop_penalty = -0.05 if action == 0 else 0.0

    # --- Component 3: Clockwise progress reward ---
    progress_reward = 0.0

    if prev_pos is not None and next_pos is not None:
        prev_idx = LOOP_INDEX.get(prev_pos)
        next_idx = LOOP_INDEX.get(next_pos)

        if prev_idx is not None and next_idx is not None:
            # Compute clockwise progress (positive = clockwise advance)
            delta = (next_idx - prev_idx) % N_WAYPOINTS

            if delta == 1:
                # Moved one step clockwise
                progress_reward = 0.3
            elif delta == N_WAYPOINTS - 1:
                # Moved one step counter-clockwise (penalize)
                progress_reward = -0.2
            # delta == 0 means stayed in same loop cell (no bonus)

        elif prev_idx is not None and next_idx is None:
            # Agent moved off the known loop path — mild penalty
            progress_reward = -0.05

        elif prev_idx is None and next_idx is not None:
            # Agent moved onto the loop path — small encouragement
            progress_reward = 0.05

    # --- Component 4: Checkpoint bonus (stepping on any marker position) ---
    checkpoint_bonus = 0.0
    if next_pos is not None and next_pos in MARKER_POSITIONS:
        # Only reward if we actually moved to the marker (not just stayed)
        if next_pos != prev_pos:
            checkpoint_bonus = 0.5

    # --- Component 5: Loop completion bonus (use env_reward if available) ---
    loop_bonus = 0.0
    if env_reward > 0:
        loop_bonus = env_reward * 5.0  # amplify sparse signal

    # --- Total reward ---
    total = (
        step_penalty
        + noop_penalty
        + progress_reward
        + checkpoint_bonus
        + loop_bonus
    )

    return {
        "total": float(total),
        "step_penalty": float(step_penalty),
        "noop_penalty": float(noop_penalty),
        "progress_reward": float(progress_reward),
        "checkpoint_bonus": float(checkpoint_bonus),
        "loop_bonus": float(loop_bonus),
        "env_reward": float(env_reward),
    }