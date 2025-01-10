import time
import gymnasium as gym
from traffic_environment import TrafficEnv
import rl_planners
import rl_agents

# define rewards function
rewards = {"state": 0}

# initialize the environment with rewards and max_steps
env = TrafficEnv(rewards = rewards, max_steps=1000)

# set the RL algorithm to plan or train an agent
rl_algo = "Value Iteration"

# initialize the agent
if rl_algo == "Value Iteration":
    agent = rl_planners.ValueIterationPlanner(env)
elif rl_algo == "Policy Iteration":
    agent = rl_planners.PolicyIterationPlanner(env)
elif rl_algo == "Q-Learning":
    agent = rl_agents.QLearningAgent(env, gamma=0.75, alpha=0.1, epsilon=1.0, epsilon_decay=0.99, episodes=2000)
    agent.train()
elif rl_algo == "SARSA":
    agent = rl_agents.SARSAgent(env, gamma=0.75, alpha=0.1, epsilon=1.0, epsilon_decay=0.99, episodes=2000)
    agent.train()

# TODO: Initialize variables to track performance metrics
# Metrics to include:
# 1. Count of instances where car count exceeds critical thresholds (N total cars or M in any direction)
# 2. Average number of cars waiting at the intersection in all directions during a time period
# 3. Maximum continuous time where car count remains below critical thresholds


# reset the environment and get the initial observation
observation, info = env.reset(seed=42), {}

# TODO: Initialize variables to track environment metrics
# Example: cumulative rewards, episode duration, etc.

# set light state variables
RED, GREEN = 0, 1

# run the environment until terminated or truncated
terminated, truncated = False, False
while (not terminated and not truncated):
    # use the agent's policy to choose an action
    action = agent.choose_action(observation)
    # step through the environment with the chosen action
    observation, reward, terminated, truncated, info = env.step(action)

    # TODO: Update variables to calculate performance and environment metrics based on the new observation

    # unpack the state to get the number of cars and traffic light state
    ns, ew, light = tuple(observation)
    light_color = "GREEN" if light == GREEN else "RED"
    # print the current state
    print(f"Step: {_}, NS Cars: {ns}, EW Cars: {ew}, Light NS: {light_color}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    # render the environment at each step
    env.render()
    # add a delay to slow down the rendering for better visualization
    time.sleep(0.8)

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
