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
    'steps_this_loop': 0,
    'initialized': False,
    'recent_positions': [],      # short history to detect oscillation
    'consecutive_noops': 0,      # track noop streaks
}

# The minimum path around the loop is 8 steps (tight clockwise circuit)
OPTIMAL_LOOP_STEPS = 8
# A reasonable loop should take no more than ~16 steps
GOOD_LOOP_STEPS = 16
# Hard ceiling — beyond this the agent is clearly wandering
MAX_LOOP_STEPS = 32

# How many recent positions to track for oscillation detection
HISTORY_LEN = 6


def _get_agent_pos(obs):
    """Find agent position (row, col) from observation grid."""
    board = obs[0]
    positions = np.argwhere(board == 2.0)
    if len(positions) > 0:
        return tuple(positions[0])
    return None


def _get_marker_at(pos):
    """Return marker name if position matches a known marker, else None."""
    return POS_TO_MARKER.get(pos, None)


def _detect_reset(next_agent_pos, state):
    """Detect episode reset via agent teleportation (Manhattan dist > 1)."""
    if state['prev_agent_pos'] is None or next_agent_pos is None:
        return False
    pr, pc = state['prev_agent_pos']
    nr, nc = next_agent_pos
    return abs(nr - pr) + abs(nc - pc) > 1


def _reset_state():
    """Reset all persistent state for a new episode."""
    _state['last_checkpoint_idx'] = None
    _state['checkpoints_hit'] = 0
    _state['loops_completed'] = 0
    _state['prev_agent_pos'] = None
    _state['steps_since_checkpoint'] = 0
    _state['steps_this_loop'] = 0
    _state['initialized'] = True
    _state['recent_positions'] = []
    _state['consecutive_noops'] = 0


def _speed_bonus(steps):
    """
    Compute speed bonus for completing a loop in `steps` steps.
    Peaks at OPTIMAL_LOOP_STEPS, linearly decays to 0 at MAX_LOOP_STEPS.
    """
    if steps <= OPTIMAL_LOOP_STEPS:
        return 6.0
    elif steps <= GOOD_LOOP_STEPS:
        frac = 1.0 - (steps - OPTIMAL_LOOP_STEPS) / (GOOD_LOOP_STEPS - OPTIMAL_LOOP_STEPS)
        return 6.0 * frac  # 6.0 → 0.0 over [8, 16]
    else:
        return 0.0


def _oscillation_penalty(recent_positions):
    """
    Penalize if the agent is oscillating (same position appears multiple
    times in recent history with very short cycle).
    """
    if len(recent_positions) < 4:
        return 0.0
    # Check for A-B-A-B pattern (length-2 cycle)
    if (recent_positions[-1] == recent_positions[-3] and
            recent_positions[-2] == recent_positions[-4]):
        return -0.15
    return 0.0


def reward_fn(prev_obs, action, next_obs, info) -> dict:
    global _state

    if not _state['initialized']:
        _reset_state()

    prev_agent_pos = _get_agent_pos(prev_obs)
    next_agent_pos = _get_agent_pos(next_obs)

    # Episode reset detection
    if _detect_reset(next_agent_pos, _state):
        _reset_state()
    if info.get('episode') is not None or info.get('reset', False):
        _reset_state()

    # ---------------------------------------------------------------
    # Reward components
    # ---------------------------------------------------------------
    checkpoint_reward    = 0.0
    loop_reward          = 0.0
    speed_bonus_val      = 0.0
    step_penalty         = 0.0
    noop_penalty         = 0.0
    wall_bump_penalty    = 0.0
    direction_bonus      = 0.0
    wrong_dir_penalty    = 0.0
    oscillation_penalty  = 0.0
    stall_penalty        = 0.0

    # --- Base step cost: every step has a cost to incentivize speed ---
    step_penalty = -0.1

    # --- NOOP penalty: strongly discourage standing still ---
    if action == 0:
        _state['consecutive_noops'] += 1
        # Escalating penalty for consecutive noops
        noop_penalty = -0.5 * _state['consecutive_noops']
        # Cap at -2.0 to avoid destabilizing training
        noop_penalty = max(noop_penalty, -2.0)
    else:
        _state['consecutive_noops'] = 0

    # --- Wall bump: agent tried to move but didn't ---
    moved = (prev_agent_pos != next_agent_pos) if (prev_agent_pos and next_agent_pos) else True
    if not moved and action != 0:
        wall_bump_penalty = -0.3

    # --- Update step counters ---
    _state['steps_this_loop'] += 1
    _state['steps_since_checkpoint'] += 1

    # --- Update position history for oscillation detection ---
    if next_agent_pos is not None:
        _state['recent_positions'].append(next_agent_pos)
        if len(_state['recent_positions']) > HISTORY_LEN:
            _state['recent_positions'].pop(0)

    # --- Oscillation penalty ---
    oscillation_penalty = _oscillation_penalty(_state['recent_positions'])

    # --- Stall penalty: extra cost when spending too long between checkpoints ---
    if _state['steps_since_checkpoint'] > GOOD_LOOP_STEPS:
        excess = _state['steps_since_checkpoint'] - GOOD_LOOP_STEPS
        stall_penalty = -0.05 * excess  # grows linearly with stall length

    # --- Checkpoint detection and clockwise ordering ---
    if next_agent_pos is not None:
        marker = _get_marker_at(next_agent_pos)

        if marker is not None:
            current_marker_idx = CLOCKWISE_INDEX[marker]

            if _state['last_checkpoint_idx'] is None:
                # First checkpoint — reward for reaching any marker
                checkpoint_reward = 1.0
                _state['last_checkpoint_idx'] = current_marker_idx
                _state['checkpoints_hit'] = 1
                _state['steps_since_checkpoint'] = 0
                # Don't reset steps_this_loop yet — wait for full loop

            elif current_marker_idx != _state['last_checkpoint_idx']:
                last_idx = _state['last_checkpoint_idx']
                expected_next_idx = (last_idx + 1) % 4

                if current_marker_idx == expected_next_idx:
                    # Correct clockwise checkpoint!
                    checkpoint_reward = 2.0
                    _state['checkpoints_hit'] += 1
                    _state['last_checkpoint_idx'] = current_marker_idx
                    _state['steps_since_checkpoint'] = 0

                    # Full loop completed every 4 correct checkpoints
                    if _state['checkpoints_hit'] % 4 == 0:
                        _state['loops_completed'] += 1
                        n = _state['loops_completed']

                        # Loop reward grows with each loop completed:
                        # gives a persistent gradient signal even at high performance
                        loop_reward = 8.0 + n * 1.0

                        # Speed bonus for this loop
                        speed_bonus_val = _speed_bonus(_state['steps_this_loop'])

                        _state['steps_this_loop'] = 0
                        # Clear oscillation history on loop completion
                        _state['recent_positions'] = []

                else:
                    # Wrong direction — significant penalty
                    wrong_dir_penalty = -3.0
                    _state['last_checkpoint_idx'] = current_marker_idx
                    _state['steps_since_checkpoint'] = 0

    # --- Potential-based direction shaping toward next checkpoint ---
    if (next_agent_pos is not None and prev_agent_pos is not None
            and _state['last_checkpoint_idx'] is not None):

        next_expected_idx = (_state['last_checkpoint_idx'] + 1) % 4
        next_cp_name = CLOCKWISE_ORDER[next_expected_idx]
        next_cp_pos = MARKER_POSITIONS[next_cp_name]

        prev_dist = (abs(prev_agent_pos[0] - next_cp_pos[0])
                     + abs(prev_agent_pos[1] - next_cp_pos[1]))
        next_dist = (abs(next_agent_pos[0] - next_cp_pos[0])
                     + abs(next_agent_pos[1] - next_cp_pos[1]))

        if next_dist < prev_dist:
            direction_bonus = 0.15   # moved closer to next checkpoint
        elif next_dist > prev_dist and action != 0:
            direction_bonus = -0.08  # moved away (not a NOOP — already penalized)

    # --- Update persistent state ---
    _state['prev_agent_pos'] = next_agent_pos

    # --- Total ---
    total = (
        checkpoint_reward
        + loop_reward
        + speed_bonus_val
        + step_penalty
        + noop_penalty
        + wall_bump_penalty
        + direction_bonus
        + wrong_dir_penalty
        + oscillation_penalty
        + stall_penalty
    )

    return {
        "total": float(total),
        "checkpoint_reward": checkpoint_reward,
        "loop_reward": loop_reward,
        "speed_bonus": speed_bonus_val,
        "step_penalty": step_penalty,
        "noop_penalty": noop_penalty,
        "wall_bump_penalty": wall_bump_penalty,
        "direction_bonus": direction_bonus,
        "wrong_dir_penalty": wrong_dir_penalty,
        "oscillation_penalty": oscillation_penalty,
        "stall_penalty": stall_penalty,
    }