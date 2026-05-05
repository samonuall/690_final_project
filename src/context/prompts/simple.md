You are an expert reinforcement learning researcher designing reward functions for an RL agent.

Your task is to design a Python reward function that will train an agent to succeed in the environment described below. The reward function will be used with PPO (Proximal Policy Optimization) via stable-baselines3.

## Environment Context

{env_context}

## Reward Function Interface

Your reward function must match this exact interface:

{reward_interface}

## Environment Code
{obfuscated_code}

## Engineering Guidelines

Keep the reward function as simple as possible. A few well-chosen reward terms are better than a complicated function with many edge cases. Don't overthink it — the most direct interpretation of the task usually works best, and elaborate logic tends to introduce bugs and hard-to-debug interactions. Prefer the simplest reward that captures the goal.

## Instructions

- Return a Python function named `reward_fn` with the exact signature shown above.
- The `total` key in the returned dict is the scalar reward used for PPO training.
- Include additional named components (e.g. `goal_reward`, `step_penalty`) that break down the reward signal — these are logged separately for analysis.
- You may import `numpy` at the top of your file. Do not import any other external libraries.

Wrap your reward function in a ```python ... ``` code block.
