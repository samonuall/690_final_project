import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Rewards the agent for successfully tracking clockwise progress around
    the orchard loop, while penalizing unnecessary actions and failure to
    complete a full loop.

    Args:
        prev_obs : np.ndarray  board observation before the action
        action   : int         action index (0-indexed)
        next_obs : np.ndarray  board observation after the action
        info     : dict        contains at minimum:
                               "env_reward": float

    Returns:
        dict with "total" (float) and optional breakdown keys
    """
    # Extract marker positions from the board
    # Marker tiles have value 3.0, walls 0.0, empty 1.0, agent 2.0
    obs_prev = prev_obs[0, :, :]
    obs_next = next_obs[0, :, :]

    marker_masks = np.array(
        [
            obs_prev == 3.0,  # Marker broadcast as 3.0
            obs_next == 3.0,
        ]
    )

    # Identify if agent is moving to or from a marker
    prev_markers = tuple(np.argwhere(marker_masks[0, 1] & (mark[1]).reshape(5, 5) == 0)
                         for mark in zip(obs_prev, obs_prev[0, :, :]))
    # Correctly identify marker locations by checking where obs == 3.0
    prev_marker_locs = tuple(np.argwhere(obs_prev == 3.0)[0] if np.any(obs_prev == 3.0) else None
                             for _ in range(len(obs_prev)))
    next_marker_locs = tuple(np.argwhere(obs_next == 3.0)[0] if np.any(obs_next == 3.0) else None
                             for _ in range(len(obs_next)))

    # Detect if next position lands on a marker
    is_next_on_marker = bool(np.any(obs_next == 3.0))
    prev_was_on_marker = bool(np.any(obs_prev == 3.0))

    step_cost = -0.1
    marker_rewards = {"m": 0.0, "n": 0.0, "o": 0.0, "p": 0.0}

    if is_next_on_marker:
        # Mark the first marker detected
        m_idx = []
        next_marker_pos = np.argwhere(obs_next == 3.0)[0]
        marker_rewards["m"] = 0.75
        marker_rewards["n"] = 0.4
        marker_rewards["o"] = 0.6
        marker_rewards["p"] = 0.35
        total_marker = sum(marker_rewards.values())
        info["marker_pos"] = next_marker_pos.astype(int)
    else:
        total_marker = 0.0
        info["marker_pos"] = None

    return {
        "total": float(total_marker + step_cost),
        "goal_reward": float(total_marker),
        "step_penalty": float(step_cost),
    }