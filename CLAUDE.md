Goal: See if LLM designed reward functions lead to misalignment, and if we can make them better. 


## Main Context
We will use stable-baselines3 to train agents using PPO with the same network and training hyperparameters.
We will use openrouter to access LLM models for designing reward functions given environment code.
We will obfuscate original env code so that LLMs are not aware of the exact environment from their pretraining knowledge.
The goal is to understand and analyze how LLMs design reward functions for RL agents in different tasks, and if they are prone to designing bad reward functions.
We will also explore if adding prompts about reward alignment or adding self reflection related to this allows LLMs to be good at designing safe and aligned reward functions. 
We will start with https://github.com/biological-alignment-benchmarks/ai-safety-gridworlds as the environment for testing LLM's capabilities for designing good reward functions. Specifcally the lava and 
boat race environments.
The LLMs designing reward functions will need to be able to do it iteratively. This means first step design reward function then get back the agent's behavior after being trained with the reward function,
then tweaking the reward functions and trying again for x iterations.

## LLM Loop
1. LLM is given obfuscated environment code and task description.
2. LLM designs a reward function based on the given information in a separate file.
3. The designed reward function is used to train an RL agent using PPO for a fixed number of iterations.
4. The trained agent's behavior is evaluated and feedback is provided to the LLM.
    - We will follow Eureka's approach (https://arxiv.org/pdf/2310.12931) to give feedback to the LLM
    - The LLM's generated reward functions will return reward components, basically a dict with multiple rewards, and total reward
    - Total reward used in PPO to train the agent, reward components are just logged during test time eval.
    - During test time every x steps, we log the reward components and the value of a hidden fitness function to give back to the LLM as feedback
5. The LLM can then tweak the reward function based on the feedback and repeat the process for a fixed number of iterations or until a satisfactory reward function is designed.

## Environments
- Feedback for LLMs will be a few trajectories that the trained agent took, and a list of (reward component, fitness score) pairs for those trajectories as well

### Lava
- Fitness function: manhatten distance to goal along with penalty for going in lava

### Boat Race
- Fitness function: number of tiles the boat goes over?
- Feedback should have a way for the LLM to tell if the agent is spamming left/right and not actually going in circles. Maybe also include state, action pairs for a couple episodes of the agent

### Lunar Lander
- Fitness function: binary yes or no for if it landed correctly along with subtracting time it took