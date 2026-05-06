import numpy as np

# Known fixed positions (row, col) in the 5x5 grid
MARKER_POSITIONS = {
    'm': (1, 3),
    'n': (2, 1),
    'o': (2, 3),
    'p': (3, 2),
}

# Clockwise order of markers
CLOCKWISE_ORDER = ['m', 'o', 'p', 'n']  # m->o->p->n->m->...
CLOCKWISE_INDEX = {name: i for i, name in enumerate(CLOCKWISE_ORDER)}

# Counter-clockwise order (wrong direction)
COUNTER_CLOCKWISE_ORDER = ['m', 'n', 'p', 'o']

# Reverse lookup: position -> marker name
POS_TO_MARKER = {v: k for k, v in MARKER_POSITIONS.items()}

# Persistent state across calls (using a mutable container)
_state = {
    'last_checkpoint_idx': None,  # index in CLOCKWISE_ORDER of last visited checkpoint
    'checkpoints_hit': 0,         # total checkpoints hit in correct order
    'loops_completed': 0,
    'prev_agent_pos': None,
    'steps_since_checkpoint': 0,
    'visited_positions': set(),   # for loop detection / exploration bonus
    'last_obs_hash': None,        # detect episode resets
}


def _get_agent_pos(obs):
    """Find agent position (row, col) from observation grid."""
    board = obs[0]  # shape (5,5)
    positions = np.argwhere(board == 2.0)
    if len(positions) > 0:
        return tuple(positions[0])
    return None


def _get_marker_at(pos):
    """Return marker name if position matches a known marker, else None."""
    return POS_TO_MARKER.get(pos, None)


def _obs_hash(obs):
    """Simple hash to detect episode resets."""
    return obs.tobytes()


def reward_fn(prev_obs, action, next_obs, info) -> dict:
    global _state

    # --- Detect episode reset by checking if agent is back at start position ---
    # Start position for agent is (1,1) based on the map layout
    START_POS = (1, 1)
    
    prev_agent_pos = _get_agent_pos(prev_obs)
    next_agent_pos = _get_agent_pos(next_obs)

    # Detect reset: if prev agent was None or the obs changed dramatically
    # We check if next agent is at start AND state looks fresh
    # Use a simple heuristic: if prev_agent_pos == START_POS and _state['prev_agent_pos'] != START_POS
    # More robustly: track last obs hash
    current_hash = _obs_hash(next_obs)
    
    # Detect episode reset by checking if agent returns to start with cleared board
    # A reset typically puts the agent back at (1,1)
    if _state['prev_agent_pos'] is None:
        # First call ever
        _state['last_checkpoint_idx'] = None
        _state['checkpoints_hit'] = 0
        _state['loops_completed'] = 0
        _state['steps_since_checkpoint'] = 0
        _state['visited_positions'] = set()
    
    # Check if this looks like a new episode (agent at start, previous state suggests reset)
    if next_agent_pos == START_POS and _state['prev_agent_pos'] is not None:
        prev_board = prev_obs[0]
        next_board = next_obs[0]
        # If previous obs didn't have agent at start, and now it does,
        # might be a reset — but we can't be sure without more info.
        # Use steps_since_checkpoint as a heuristic: if it's been very long, consider partial reset.
        pass

    # Better reset detection: use info dict if available
    # If 'episode' key in info, we know it's a fresh start
    if info.get('episode') is not None or info.get('reset', False):
        _state['last_checkpoint_idx'] = None
        _state['checkpoints_hit'] = 0
        _state['loops_completed'] = 0
        _state['steps_since_checkpoint'] = 0
        _state['visited_positions'] = set()

    # Also reset state if agent teleports (non-adjacent move = episode reset)
    if _state['prev_agent_pos'] is not None and prev_agent_pos is not None:
        pr, pc = _state['prev_agent_pos']
        if next_agent_pos is not None:
            nr, nc = next_agent_pos
            dist = abs(nr - pr) + abs(nc - pc)
            if dist > 1:
                # Teleport detected — episode reset
                _state['last_checkpoint_idx'] = None
                _state['checkpoints_hit'] = 0
                _state['loops_completed'] = 0
                _state['steps_since_checkpoint'] = 0
                _state['visited_positions'] = set()

    # Initialize rewards
    checkpoint_reward = 0.0
    loop_reward = 0.0
    step_penalty = 0.0
    noop_penalty = 0.0
    wall_bump_penalty = 0.0
    direction_bonus = 0.0
    anti_repeat_penalty = 0.0

    # --- Step penalty to encourage efficiency ---
    step_penalty = -0.01

    # --- NOOP penalty ---
    if action == 0:
        noop_penalty = -0.05

    # --- Wall bump penalty (agent didn't move) ---
    if prev_agent_pos is not None and next_agent_pos is not None:
        if prev_agent_pos == next_agent_pos and action != 0:
            wall_bump_penalty = -0.1

    # --- Checkpoint detection ---
    if next_agent_pos is not None:
        marker = _get_marker_at(next_agent_pos)
        
        if marker is not None:
            current_marker_idx = CLOCKWISE_INDEX[marker]
            
            if _state['last_checkpoint_idx'] is None:
                # First checkpoint — reward regardless of which one
                checkpoint_reward = 1.0
                _state['last_checkpoint_idx'] = current_marker_idx
                _state['checkpoints_hit'] = 1
                _state['steps_since_checkpoint'] = 0
            else:
                last_idx = _state['last_checkpoint_idx']
                
                # Check if this is a different checkpoint than the last
                if current_marker_idx != last_idx:
                    # Check clockwise: next expected = (last_idx + 1) % 4
                    expected_next_idx = (last_idx + 1) % 4
                    
                    if current_marker_idx == expected_next_idx:
                        # Correct clockwise direction!
                        checkpoint_reward = 2.0
                        _state['checkpoints_hit'] += 1
                        _state['last_checkpoint_idx'] = current_marker_idx
                        _state['steps_since_checkpoint'] = 0
                        
                        # Bonus for completing a full loop (every 4 checkpoints)
                        if _state['checkpoints_hit'] % 4 == 0:
                            loop_reward = 5.0
                            _state['loops_completed'] += 1
                    else:
                        # Wrong direction (counter-clockwise or skipped)
                        checkpoint_reward = -1.0
                        # Update position but don't count as valid
                        _state['last_checkpoint_idx'] = current_marker_idx
                        _state['steps_since_checkpoint'] = 0
        else:
            _state['steps_since_checkpoint'] += 1

    # --- Direction bonus: reward moving in the clockwise direction ---
    # Clockwise motion: generally moving right on top, down on right, left on bottom, up on left
    # We can give a small bonus for moving toward the next expected checkpoint
    if (next_agent_pos is not None and prev_agent_pos is not None 
            and _state['last_checkpoint_idx'] is not None):
        
        next_expected_idx = (_state['last_checkpoint_idx'] + 1) % 4
        next_checkpoint_name = CLOCKWISE_ORDER[next_expected_idx]
        next_checkpoint_pos = MARKER_POSITIONS[next_checkpoint_name]
        
        # Distance from prev to target vs next to target
        prev_dist = abs(prev_agent_pos[0] - next_checkpoint_pos[0]) + abs(prev_agent_pos[1] - next_checkpoint_pos[1])
        next_dist = abs(next_agent_pos[0] - next_checkpoint_pos[0]) + abs(next_agent_pos[1] - next_checkpoint_pos[1])
        
        if next_dist < prev_dist:
            direction_bonus = 0.05
        elif next_dist > prev_dist and action != 0:
            direction_bonus = -0.02

    # --- Anti-repetition: penalize revisiting same position too much ---
    if next_agent_pos is not None:
        pos_key = next_agent_pos
        if pos_key in _state['visited_positions']:
            anti_repeat_penalty = -0.005  # very small, don't over-penalize
        _state['visited_positions'].add(pos_key)
        # Keep visited set from growing unboundedly (reset on loop completion)
        if _state['checkpoints_hit'] % 4 == 0 and _state['checkpoints_hit'] > 0:
            _state['visited_positions'] = set()

    # --- Update state ---
    _state['prev_agent_pos'] = next_agent_pos

    # --- Use environment reward as a signal too ---
    env_reward = info.get('env_reward', 0.0)
    env_signal = env_reward * 0.5  # scale down

    # --- Total ---
    total = (
        checkpoint_reward
        + loop_reward
        + step_penalty
        + noop_penalty
        + wall_bump_penalty
        + direction_bonus
        + anti_repeat_penalty
        + env_signal
    )

    return {
        "total": total,
        "checkpoint_reward": checkpoint_reward,
        "loop_reward": loop_reward,
        "step_penalty": step_penalty,
        "noop_penalty": noop_penalty,
        "wall_bump_penalty": wall_bump_penalty,
        "direction_bonus": direction_bonus,
        "anti_repeat_penalty": anti_repeat_penalty,
        "env_signal": env_signal,
    }