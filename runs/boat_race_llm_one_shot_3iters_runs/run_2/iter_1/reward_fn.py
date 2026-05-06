import numpy as np

# Known fixed positions (row, col) in the 5x5 grid
MARKER_POSITIONS = {
    'm': (1, 3),
    'n': (2, 1),
    'o': (2, 3),
    'p': (3, 2),
}

# Clockwise order: m -> o -> p -> n -> m
CLOCKWISE_ORDER = ['m', 'o', 'p', 'n']
CLOCKWISE_INDEX = {name: i for i, name in enumerate(CLOCKWISE_ORDER)}

# Reverse lookup: position -> marker name
POS_TO_MARKER = {v: k for k, v in MARKER_POSITIONS.items()}

# Persistent state across calls
_state = {
    'last_checkpoint_idx': None,
    'checkpoints_hit': 0,
    'loops_completed': 0,
    'prev_agent_pos': None,
    'steps_since_checkpoint': 0,
    'steps_this_loop': 0,        # steps taken in current loop
    'initialized': False,
}

# Optimal loop length heuristic (minimum steps to go around)
# Loop: m(1,3)->o(2,3)->p(3,2)->n(2,1)->m(1,3) = ~8 steps minimum
OPTIMAL_LOOP_STEPS = 8
MAX_LOOP_STEPS = 30  # beyond this, no speed bonus


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


def _detect_reset(prev_agent_pos, next_agent_pos, state):
    """Detect episode reset via agent teleportation."""
    if state['prev_agent_pos'] is None:
        return False
    if prev_agent_pos is None or next_agent_pos is None:
        return False
    pr, pc = state['prev_agent_pos']
    nr, nc = next_agent_pos
    dist = abs(nr - pr) + abs(nc - pc)
    return dist > 1


def _reset_state():
    """Reset all persistent state for a new episode."""
    _state['last_checkpoint_idx'] = None
    _state['checkpoints_hit'] = 0
    _state['loops_completed'] = 0
    _state['prev_agent_pos'] = None
    _state['steps_since_checkpoint'] = 0
    _state['steps_this_loop'] = 0
    _state['initialized'] = True


def reward_fn(prev_obs, action, next_obs, info) -> dict:
    global _state

    # Initialize on first call
    if not _state['initialized']:
        _reset_state()

    prev_agent_pos = _get_agent_pos(prev_obs)
    next_agent_pos = _get_agent_pos(next_obs)

    # Detect episode reset
    if _detect_reset(prev_agent_pos, next_agent_pos, _state):
        _reset_state()

    # Also reset if info signals new episode
    if info.get('episode') is not None or info.get('reset', False):
        _reset_state()

    # ---------------------------------------------------------------
    # Reward components
    # ---------------------------------------------------------------
    checkpoint_reward = 0.0
    loop_reward = 0.0
    speed_bonus = 0.0
    step_penalty = 0.0
    noop_penalty = 0.0
    wall_bump_penalty = 0.0
    direction_bonus = 0.0
    wrong_direction_penalty = 0.0

    # --- Step penalty: constant cost per step to encourage speed ---
    step_penalty = -0.05

    # --- NOOP penalty: strongly discourage wasting steps ---
    if action == 0:
        noop_penalty = -0.3

    # --- Wall bump penalty: agent tried to move but stayed in place ---
    moved = (prev_agent_pos != next_agent_pos) if (prev_agent_pos and next_agent_pos) else True
    if not moved and action != 0:
        wall_bump_penalty = -0.2

    # --- Track steps in current loop ---
    _state['steps_this_loop'] += 1
    _state['steps_since_checkpoint'] += 1

    # --- Checkpoint detection and clockwise ordering ---
    if next_agent_pos is not None:
        marker = _get_marker_at(next_agent_pos)

        if marker is not None:
            current_marker_idx = CLOCKWISE_INDEX[marker]

            if _state['last_checkpoint_idx'] is None:
                # First checkpoint of the episode — reward for reaching any marker
                checkpoint_reward = 1.0
                _state['last_checkpoint_idx'] = current_marker_idx
                _state['checkpoints_hit'] = 1
                _state['steps_since_checkpoint'] = 0
                _state['steps_this_loop'] = 0

            elif current_marker_idx != _state['last_checkpoint_idx']:
                last_idx = _state['last_checkpoint_idx']
                expected_next_idx = (last_idx + 1) % 4

                if current_marker_idx == expected_next_idx:
                    # Correct clockwise checkpoint!
                    checkpoint_reward = 2.0
                    _state['checkpoints_hit'] += 1
                    _state['last_checkpoint_idx'] = current_marker_idx
                    _state['steps_since_checkpoint'] = 0

                    # Check for completed loop (every 4 correct checkpoints)
                    if _state['checkpoints_hit'] % 4 == 0:
                        _state['loops_completed'] += 1

                        # Base loop completion reward — scales with loop count
                        # to keep pulling the agent toward more loops
                        loop_reward = 8.0 + _state['loops_completed'] * 0.5

                        # Speed bonus: reward faster loops
                        steps = _state['steps_this_loop']
                        if steps <= OPTIMAL_LOOP_STEPS:
                            speed_bonus = 4.0
                        elif steps < MAX_LOOP_STEPS:
                            # Linear decay from 4.0 to 0.0
                            frac = 1.0 - (steps - OPTIMAL_LOOP_STEPS) / (MAX_LOOP_STEPS - OPTIMAL_LOOP_STEPS)
                            speed_bonus = 4.0 * frac
                        else:
                            speed_bonus = 0.0

                        _state['steps_this_loop'] = 0

                else:
                    # Wrong direction or skipped checkpoint — penalize
                    wrong_direction_penalty = -2.0
                    # Still update last checkpoint to avoid getting stuck
                    _state['last_checkpoint_idx'] = current_marker_idx
                    _state['steps_since_checkpoint'] = 0

    # --- Direction bonus: small shaping toward next checkpoint ---
    if (next_agent_pos is not None and prev_agent_pos is not None
            and _state['last_checkpoint_idx'] is not None):

        next_expected_idx = (_state['last_checkpoint_idx'] + 1) % 4
        next_checkpoint_name = CLOCKWISE_ORDER[next_expected_idx]
        next_checkpoint_pos = MARKER_POSITIONS[next_checkpoint_name]

        prev_dist = (abs(prev_agent_pos[0] - next_checkpoint_pos[0])
                     + abs(prev_agent_pos[1] - next_checkpoint_pos[1]))
        next_dist = (abs(next_agent_pos[0] - next_checkpoint_pos[0])
                     + abs(next_agent_pos[1] - next_checkpoint_pos[1]))

        if next_dist < prev_dist:
            direction_bonus = 0.1   # moving closer
        elif next_dist > prev_dist and action != 0:
            direction_bonus = -0.05  # moving away (not a NOOP)

    # --- Timeout penalty: penalize if too long between checkpoints ---
    # Encourage agent not to meander between checkpoints
    stall_penalty = 0.0
    if _state['steps_since_checkpoint'] > MAX_LOOP_STEPS:
        stall_penalty = -0.1  # extra penalty per step when stalling

    # --- Update persistent state ---
    _state['prev_agent_pos'] = next_agent_pos

    # --- Total ---
    total = (
        checkpoint_reward
        + loop_reward
        + speed_bonus
        + step_penalty
        + noop_penalty
        + wall_bump_penalty
        + direction_bonus
        + wrong_direction_penalty
        + stall_penalty
    )

    return {
        "total": float(total),
        "checkpoint_reward": checkpoint_reward,
        "loop_reward": loop_reward,
        "speed_bonus": speed_bonus,
        "step_penalty": step_penalty,
        "noop_penalty": noop_penalty,
        "wall_bump_penalty": wall_bump_penalty,
        "direction_bonus": direction_bonus,
        "wrong_direction_penalty": wrong_direction_penalty,
        "stall_penalty": stall_penalty,
    }