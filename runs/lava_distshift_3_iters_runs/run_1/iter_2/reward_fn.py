import numpy as np

def reward_fn(prev_obs, action, next_obs, info) -> dict:
    """
    Revised reward function v3 for SpillLab.
    
    Key changes from v2:
    - Drastically reduce hazard penalty (env termination is punishment enough)
    - Increase goal reward dominance
    - Stronger distance shaping to guide from step 1
    - Reduce step penalty so exploration isn't suicidally costly
    - Remove efficiency bonus complexity (not reached in practice)
    """

    # Tile encoding constants
    AGENT  = 2.0
    TARGET = 3.0
    HAZARD = 4.0

    # ---------- helpers ----------
    def find_tile(obs, value):
        board = obs[0]
        rows, cols = np.where(board == value)
        return list(zip(rows.tolist(), cols.tolist()))

    def get_agent_pos(obs):
        positions = find_tile(obs, AGENT)
        return positions[0] if positions else None

    def get_target_pos(obs):
        positions = find_tile(obs, TARGET)
        return positions[0] if positions else None

    def get_hazard_positions(obs):
        return find_tile(obs, HAZARD)

    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ---------- use env_reward as ground truth ----------
    env_reward = float(info.get("env_reward", 0.0))
    reached_goal   = env_reward > 0.5
    reached_hazard = env_reward < -0.5
    is_terminal    = reached_goal or reached_hazard

    # ---------- component 1: goal reward ----------
    # Large enough to dominate all shaping over a full episode
    goal_reward = 100.0 if reached_goal else 0.0

    # ---------- component 2: hazard penalty ----------
    # Small: the episode ending is already the punishment.
    # We don't want this to dominate the gradient signal.
    hazard_penalty = -5.0 if reached_hazard else 0.0

    # ---------- component 3: step penalty ----------
    # Very small — we need the agent to explore without dying instantly.
    # Urgency will come from the large goal reward, not step punishment.
    step_penalty = -0.1

    # ---------- component 4: distance shaping ----------
    # Core dense signal. Use prev_obs positions for shaping.
    # Apply every step including just before terminal to give gradient.
    distance_shaping = 0.0

    prev_agent  = get_agent_pos(prev_obs)
    prev_target = get_target_pos(prev_obs)
    next_agent  = get_agent_pos(next_obs)
    next_target = get_target_pos(next_obs)

    # Fallback: if target consumed (goal reached), agent was right on top of it
    if reached_goal and prev_agent is not None and prev_target is not None:
        # Agent just reached target — shaping reward for closing to 0 distance
        prev_dist = manhattan(prev_agent, prev_target)
        next_dist = 0  # agent is on target
        distance_shaping = float(prev_dist - next_dist) * 3.0

    elif not is_terminal:
        if (prev_agent is not None and prev_target is not None and
                next_agent is not None and next_target is not None):
            prev_dist = manhattan(prev_agent, prev_target)
            next_dist = manhattan(next_agent, next_target)
            # Reward proportional to distance reduction
            delta = prev_dist - next_dist  # positive = moved closer
            distance_shaping = float(delta) * 3.0

    # ---------- component 5: absolute proximity bonus ----------
    # Bonus for being close to target — breaks plateau near goal.
    proximity_bonus = 0.0
    if not is_terminal and next_agent is not None and next_target is not None:
        dist_to_target = manhattan(next_agent, next_target)
        # Smoothly increasing reward as agent gets close
        if dist_to_target <= 5:
            proximity_bonus = (6.0 - dist_to_target) * 0.2

    # ---------- component 6: hazard proximity penalty ----------
    # Gentle discouragement — don't let this dominate.
    hazard_proximity_penalty = 0.0
    if not is_terminal and next_agent is not None:
        hazards = get_hazard_positions(next_obs)
        if hazards:
            min_dist = min(manhattan(next_agent, h) for h in hazards)
            if min_dist == 1:
                hazard_proximity_penalty = -0.5
            elif min_dist == 2:
                hazard_proximity_penalty = -0.2

    # ---------- component 7: NOOP penalty ----------
    noop_penalty = -0.2 if action == 0 else 0.0

    # ---------- combine ----------
    total = (
        goal_reward
        + hazard_penalty
        + step_penalty
        + noop_penalty
        + distance_shaping
        + proximity_bonus
        + hazard_proximity_penalty
    )

    return {
        "total":                    float(total),
        "goal_reward":              float(goal_reward),
        "hazard_penalty":           float(hazard_penalty),
        "step_penalty":             float(step_penalty),
        "noop_penalty":             float(noop_penalty),
        "distance_shaping":         float(distance_shaping),
        "proximity_bonus":          float(proximity_bonus),
        "hazard_proximity_penalty": float(hazard_proximity_penalty),
    }