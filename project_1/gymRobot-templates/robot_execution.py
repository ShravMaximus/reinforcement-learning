import time
import gymnasium as gym
from robot_environment import RobotEnv
import rl_agents
import pickle
import pandas as pd
# define rewards function
rewards = {"state": 0}

# initialize the environment with rewards and max_steps
env = RobotEnv(rewards = rewards, max_steps=1000)

# set the RL algorithm used to train the agent
rl_algo = "Q-Learning"

# initialize the agent
if rl_algo == "Q-Learning":
    agent = rl_agents.QLearningAgent(env, gamma=0.75, alpha=0.3, epsilon=1.0, epsilon_decay=0.99, episodes=2000)
elif rl_algo == "SARSA":
    agent = rl_agents.SARSAgent(env, gamma=0.75, alpha=0.1, epsilon=1.0, epsilon_decay=0.99, episodes=2000)
# train the agent
agent.train()

#
# def save_agent(agent, filename):
#
#     agent_state = {
#         'q_table': agent.q_table,
#         'gamma': agent.gamma,
#         'alpha': agent.alpha,
#         'epsilon': agent.epsilon,
#         'epsilon_decay': agent.epsilon_decay,
#         'episodes': agent.episodes
#     }
#
#     with open(filename, "wb") as f:
#         pickle.dump(agent_state, f)
#
#     print(f"Agent model saved as {filename}")
#
#
#
# def load_agent(filename, env, rl_algo="SARSA"):
#     with open(filename, "rb") as f:
#         agent_state = pickle.load(f)
#
#
#     if rl_algo == "Q-Learning":
#         agent = rl_agents.QLearningAgent(env, gamma=agent_state['gamma'], alpha=agent_state['alpha'],
#                                         epsilon=agent_state['epsilon'], epsilon_decay=agent_state['epsilon_decay'],
#                                         episodes=agent_state['episodes'])
#     elif rl_algo == "SARSA":
#         agent = rl_agents.SARSAgent(env, gamma=agent_state['gamma'], alpha=agent_state['alpha'],
#                                     epsilon=agent_state['epsilon'], epsilon_decay=agent_state['epsilon_decay'],
#                                     episodes=agent_state['episodes'])
#
#     # restore the q_table
#     agent.q_table = agent_state['q_table']
#
#     print(f"model loaded from {filename}")
#     return agent
#
#
# model_filename = f"{rl_algo}_robot_model.pkl"
# save_agent(agent,model_filename)



#agent = load_agent(f"{rl_algo}_robot_model.pkl", env, rl_algo="SARSA")  # or Q-Learning


# TODO: Initialize variables to track performance metrics
# Metrics to include:
num_box_success=0  # 1. Number of boxes successfully delivered
num_human_collisions=0   # 2. Count of harmful actions or collisions
num_non_human_collisions=0
max_safe_dur=0 # 3. Longest uninterrupted safe duration
safe_dur_cntr=0

# reset the environment and get the initial observation
observation, info = env.reset(seed=42), {}

# TODO: Initialize variables to track environment metrics
# Example: cumulative rewards, episode duration, etc.
cumulative_rewards=0
episode_durations =0

episode_counter = 0
max_episodes = 100
all_episode_metrics = []


# run the environment until terminated or truncated
terminated, truncated = False, False
while (not terminated and not truncated) and episode_counter<=max_episodes:
    agent.epsilon = 0
    # use the agent's policy to choose an action
    action = agent.choose_action(observation)
    # step through the environment with the chosen action
    observation, reward, terminated, truncated, oth = env.step(action)

    # TODO: Update variables to calculate performance and environment metrics based on the new observation
    bumped = oth.get('bumped_status', "")
    #print("bumped_status",bumped)
    if bumped == "target":
        #print("bumped")
        #time.sleep(2)
        num_box_success += 1
        safe_dur_cntr+=1
    elif bumped == "":
        safe_dur_cntr += 1
    elif bumped == "worker":
        #num_human_collisions += 1
        safe_dur_cntr=0
    elif bumped in ["wall","equipment"]:
        num_non_human_collisions += 1
        safe_dur_cntr = 0

    max_safe_dur = max(max_safe_dur, safe_dur_cntr)

    cumulative_rewards += reward
    episode_durations+=1

    # print the current state
    #print(f"Step: {episode_durations}, Observation: {observation}, Reward: {cumulative_rewards}, Terminated: {terminated}, Truncated: {truncated}")
    # render the environment at each step
    env.render()
    # Add a delay to slow down the simulation for better visualization
    #time.sleep(0.8)

    # reset the environment if terminated or truncated
    if terminated or truncated:
        print("\nTERMINATED OR TRUNCATED, RESETTING...\n")
        #time.sleep(2)
        if terminated==True:
            num_human_collisions += 1

        episode_counter += 1
        # TODO: Update metrics for completed episode
        print(f"Episode ended: Boxes delivered = {num_box_success}, Cumm Reward: {cumulative_rewards},Episode duration: {episode_durations},Human Collisions = {num_human_collisions}, Non-Human Collisions = {num_non_human_collisions},Longest Safe Duration = {max_safe_dur}")

        all_episode_metrics.append({
            "Boxes_delivered": num_box_success,
            "cumm_reward": cumulative_rewards,
            "num_steps": episode_durations,
            "num_human_collisions": num_human_collisions,
            "num_non_human_collisions": num_non_human_collisions,
            "max_safe_dur": max_safe_dur
        })



        observation, info = env.reset(), {}
        # TODO: Reset tracking variables for the new episode
        safe_dur_cntr = 0
        episode_durations = 0
        cumulative_rewards = 0
        num_box_success = 0
        num_human_collisions = 0
        num_non_human_collisions = 0
        max_safe_dur = 0

        terminated, truncated = False, False

print("Episodes Over ",episode_counter)
# close the environment
env.render(close=True)

# TODO: Evaluate performance based on high-level metrics

df = pd.DataFrame(all_episode_metrics)

# Write the DataFrame to a CSV file
df.to_csv(rl_algo+"_robot_metrics.csv", index=False)

