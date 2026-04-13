"""
Smoke test: validate both gridworld environments work with random actions.

Run with:  uv run python tests/test_envs.py
"""

import sys
import numpy as np

from src.envs.gridworld_wrapper import LLMRewardGridworld, boat_race_fitness, lava_fitness


def run_random_episode(env, max_steps=200):
    """Run one episode with random actions, return a summary dict."""
    obs, info = env.reset()
    assert obs is not None, "reset() returned None obs"
    assert obs.shape == env.observation_space.shape, (
        f"obs shape {obs.shape} != observation_space.shape {env.observation_space.shape}"
    )

    steps = 0
    total_reward = 0.0
    total_fitness = 0.0
    unique_obs = set()
    done = False

    while not done and steps < max_steps:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        assert next_obs.shape == obs.shape, (
            f"step obs shape changed: {next_obs.shape} vs {obs.shape}"
        )
        assert "reward_components" in info, "info missing 'reward_components'"
        assert "fitness" in info, "info missing 'fitness'"
        assert "total" in info["reward_components"], "reward_components missing 'total' key"
        assert isinstance(info["fitness"], float), f"fitness is not float: {type(info['fitness'])}"

        total_reward += reward
        total_fitness += info["fitness"]
        # Track unique obs values to sanity-check the agent is moving
        unique_obs.add(tuple(next_obs.flatten().tolist()))
        obs = next_obs
        done = terminated or truncated
        steps += 1

    return {
        "steps": steps,
        "terminated": terminated,
        "truncated": truncated,
        "total_reward": total_reward,
        "total_fitness": total_fitness,
        "unique_obs_count": len(unique_obs),
    }


def test_env(env_name, fitness_fn, num_episodes=3):
    print(f"\n{'='*55}")
    print(f"  Testing: {env_name}")
    print(f"{'='*55}")

    env = LLMRewardGridworld(
        env_name=env_name,
        reward_fn=lambda obs, a, nobs, info: {"total": fitness_fn(obs, a, nobs, info)},
        fitness_fn=fitness_fn,
    )

    print(f"  obs_space : {env.observation_space}")
    print(f"  act_space : {env.action_space}")

    all_ok = True
    for ep in range(num_episodes):
        try:
            result = run_random_episode(env)
            status = "OK"
        except AssertionError as e:
            result = {}
            status = f"FAIL: {e}"
            all_ok = False

        if result:
            print(
                f"  ep {ep+1}: steps={result['steps']:3d}  "
                f"reward={result['total_reward']:7.2f}  "
                f"fitness={result['total_fitness']:7.2f}  "
                f"unique_obs={result['unique_obs_count']}  "
                f"[{status}]"
            )
        else:
            print(f"  ep {ep+1}: [{status}]")

    env.close()
    return all_ok


def main():
    results = {}

    results["boat_race"] = test_env("boat_race", boat_race_fitness)
    results["distributional_shift"] = test_env("distributional_shift", lava_fitness)

    print(f"\n{'='*55}")
    print("  Summary")
    print(f"{'='*55}")
    all_passed = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name:<25} {status}")
        all_passed = all_passed and ok

    print()
    if all_passed:
        print("  All checks passed.")
        sys.exit(0)
    else:
        print("  Some checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
