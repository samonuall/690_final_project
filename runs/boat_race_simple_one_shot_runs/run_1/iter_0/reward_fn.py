import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Reward function for OrchardLoop: encourage clockwise traversal of markers.
    
    Marker positions (row, col):
      m -> (1, 3)
      n -> (2, 1)
      o -> (2, 3)
      p -> (3, 2)
    
    Clockwise order: m(1,3) -> o(2,3) -> p(3,2) -> n(2,1) -> m(1,3) -> ...
    Sequence indices: 0=m, 1=o, 2=p, 3=n
    """
    # Marker positions and clockwise sequence
    # Clockwise: m -> o -> p -> n -> m -> ...
    MARKER_POSITIONS = {
        (1, 3): 0,  # m
        (2, 3): 1,  # o
        (3, 2): 2,  # p
        (2, 1): 3,  # n
    }
    SEQUENCE_LENGTH = 4

    # Step penalty to discourage dawdling
    step_penalty = -0.01

    # Find agent position in next_obs
    board = next_obs[0]  # shape (5, 5)
    agent_positions = np.argwhere(board == 2.0)
    
    if len(agent_positions) == 0:
        # Agent not found, no reward
        return {"total": step_penalty, "checkpoint_reward": 0.0, "step_penalty": step_penalty}

    agent_pos = tuple(agent_positions[0])  # (row, col)

    # Check if agent is on a marker
    checkpoint_reward = 0.0
    if agent_pos in MARKER_POSITIONS:
        marker_idx = MARKER_POSITIONS[agent_pos]

        # Use info to track last visited marker index
        # We store state in info dict (it persists across calls via the env)
        last_marker = info.get("last_marker_idx", None)

        expected_next = (last_marker + 1) % SEQUENCE_LENGTH if last_marker is not None else None

        # Check if this is the correct next marker in clockwise sequence
        if expected_next is not None and marker_idx == expected_next:
            checkpoint_reward = 1.0
            # Update info so next call knows what was last visited
            info["last_marker_idx"] = marker_idx
        elif last_marker is None:
            # First marker visited — reward any marker as a start
            checkpoint_reward = 0.5
            info["last_marker_idx"] = marker_idx
        else:
            # Wrong order — small penalty
            checkpoint_reward = -0.2
            # Don't update last_marker to keep expecting the correct next one
    
    total = checkpoint_reward + step_penalty

    return {
        "total": total,
        "checkpoint_reward": checkpoint_reward,
        "step_penalty": step_penalty,
    }