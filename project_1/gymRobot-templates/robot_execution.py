import time
import gymnasium as gym
from robot_environment import RobotEnv
import rl_agents

# define rewards function
rewards = {"state": 0}

# initialize the environment with rewards and max_steps
env = RobotEnv(rewards = rewards, max_steps=1000)

# set the RL algorithm used to train the agent
rl_algo = "Q-Learning"

# initialize the agent
if rl_algo == "Q-Learning":
    agent = rl_agents.QLearningAgent(env, gamma=0.75, alpha=0.1, epsilon=1.0, epsilon_decay=0.99, episodes=2000)
elif rl_algo == "SARSA":
    agent = rl_agents.SARSAgent(env, gamma=0.75, alpha=0.1, epsilon=1.0, epsilon_decay=0.99, episodes=2000)
# train the agent
agent.train()

# TODO: Initialize variables to track performance metrics
# Metrics to include:
# 1. Number of boxes successfully delivered
# 2. Count of harmful actions or collisions
# 3. Longest uninterrupted safe duration

# reset the environment and get the initial observation
observation, info = env.reset(seed=42), {}

# TODO: Initialize variables to track environment metrics
# Example: cumulative rewards, episode duration, etc.

# run the environment until terminated or truncated
terminated, truncated = False, False
while (not terminated and not truncated):
    # use the agent's policy to choose an action
    action = agent.choose_action(observation)
    # step through the environment with the chosen action
    observation, reward, terminated, truncated, info = env.step(action)

    # TODO: Update variables to calculate performance and environment metrics based on the new observation

    # print the current state
    print(f"Step: {_}, Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    # render the environment at each step
    env.render()
    # Add a delay to slow down the simulation for better visualization
    time.sleep(0.1)

    # reset the environment if terminated or truncated
    if terminated or truncated:
        print("\nTERMINATED OR TRUNCATED, RESETTING...\n")

        # TODO: Update metrics for completed episode

        observation, info = env.reset(), {}

        # TODO: Reset tracking variables for the new episode

        terminated, truncated = False, False

# close the environment
env.render(close=True)

# TODO: Evaluate performance based on high-level metrics

print("\n=== PERFORMANCE EVALUATION ===")
# TODO: Print performance metrics
