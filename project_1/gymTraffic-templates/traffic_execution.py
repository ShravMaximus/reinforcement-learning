import time
import gymnasium as gym
from traffic_environment import TrafficEnv
import rl_planners
import rl_agents
import pandas as pd

for algo in ["Value Iteration","Policy Iteration","Q-Learning","SARSA"]: #add "Value Iteration",

    rl_algo = algo

    # define rewards function
    rewards = {"state": 0}

    # initialize the environment with rewards and max_steps
    env = TrafficEnv(rewards = rewards, max_steps=1000)


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
    num_instance_N=0
    num_instances_M=0
    # 2. Average number of cars waiting at the intersection in all directions during a time period
    num_cars_waiting=0
    num_steps_per_episode=0
    # 3. Maximum continuous time where car count remains below critical thresholds
    cont_time_below_threshold=0
    num_intersection_cleared=0
    max_cont_time_below_threshold=0

    all_episode_metrics = []

    # Counter to limit to 100 episodes
    episode_counter = 0
    max_episodes = 100

    # reset the environment and get the initial observation
    observation, info = env.reset(seed=42), {}

    # TODO: Initialize variables to track environment metrics
    # Example: cumulative rewards, episode duration, etc.
    cumm_reward=0
    episode_duration=0

    # set light state variables
    RED, GREEN = 0, 1

    # run the environment until terminated or truncated
    terminated, truncated = False, False
    while (not terminated and not truncated) and episode_counter<=max_episodes:
        # use the agent's policy to choose an action
        agent.epsilon = 0
        action = agent.choose_action(observation)
        # step through the environment with the chosen action
        observation, reward, terminated, truncated, info = env.step(action)
        num_steps_per_episode+=1
        cumm_reward+=reward

        # TODO: Update variables to calculate performance and environment metrics based on the new observation
        ns,ew,light=observation[0],observation[1],observation[2]
        num_cars_waiting += (ns + ew)
        if ns > env.max_cars_dir or ew > env.max_cars_dir:
            num_instances_M += 1
        if ns+ew > env.max_cars_total:
            num_instance_N += 1

        if ns < env.max_cars_dir and ew < env.max_cars_dir and (ns+ew) <env.max_cars_total:
            cont_time_below_threshold+=1
        else:
            cont_time_below_threshold=0

        max_cont_time_below_threshold=max(max_cont_time_below_threshold,cont_time_below_threshold)

            # unpack the state to get the number of cars and traffic light state
        ns, ew, light = tuple(observation)
        light_color = "GREEN" if light == GREEN else "RED"
        # print the current state
        #print(f"Step: {num_steps_per_episode}, NS Cars: {ns}, EW Cars: {ew}, Light NS: {light_color}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        # render the environment at each step
        env.render()
        # add a delay to slow down the rendering for better visualization
        #time.sleep(0.8)

        # reset the environment if terminated or truncated
        if terminated or truncated:
            print("\nTERMINATED OR TRUNCATED, RESETTING...\n")
            if terminated==True:
                num_intersection_cleared+=1

            # TODO: Update metrics for completed episode
            avg_cars_waiting = num_cars_waiting / num_steps_per_episode
            episode_duration += 1
            episode_counter += 1

            print(f"Episode Environment Metrics: ")
            print(f"  1. Cumulative Reward: {cumm_reward}")
            print(f"  2. Num Steps per Episode: {num_steps_per_episode}")
            print(f"  3. Exceed M threshold: {num_instances_M},Exceed N threshold: {num_instance_N}")
            print(f"  4. Average Number of Cars Waiting: {avg_cars_waiting}")
            print(f"  5. Maximum continuous time where car count stayed below thresholds: {max_cont_time_below_threshold}")
            print(f"  6. Num of instances intersection cleared: {num_intersection_cleared}")
            #time.sleep(10)
            all_episode_metrics.append({
                "episode": episode_counter,
                "cumm_reward": cumm_reward,
                "num_steps": num_steps_per_episode,
                "num_instances_M": num_instances_M,
                "num_instance_N": num_instance_N,
                "avg_cars_waiting": avg_cars_waiting,
                "max_cont_time_below_threshold": max_cont_time_below_threshold,
                "num_intersection_cleared": num_intersection_cleared
            })
            observation, info = env.reset(), {}

            # TODO: Reset tracking variables for the new episode

            num_instance_N = 0
            num_instances_M = 0
            num_cars_waiting = 0
            num_steps_per_episode = 0
            cumm_reward = 0
            cont_time_below_threshold=0
            max_cont_time_below_threshold=0
            terminated, truncated = False, False

    print("Episodes Over ",episode_counter)
    # close the environment
    env.render(close=True)

    # TODO: Evaluate performance based on high-level metrics

    print("\n=== PERFORMANCE EVALUATION ===")
    df = pd.DataFrame(all_episode_metrics)

    # Write the DataFrame to a CSV file
    df.to_csv(rl_algo+"_traffic_metrics.csv", index=False)
