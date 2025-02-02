import os
#os.chdir('gymTraffic-templates')
import time
import gymnasium as gym
from traffic_environment import TrafficEnv
import rl_planners
import rl_agents
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# define rewards function
rewards = {"state": 0}

# Define functions to run experiments on different algorithms

def run_sensitivity_experiment(env, gamma_values, theta_values, epsilon_values, algorithm):
    print("algorithm-",algorithm)
    results = []
    for gamma in gamma_values:
        for theta in theta_values:
            for epsilon in epsilon_values:
                # Initialize the agent with the given parameters
                if algorithm == "VI":
                    start_time = time.time()
                    agent = rl_planners.ValueIterationPlanner(env, gamma=gamma, theta=theta)
                elif algorithm == "PI":
                    start_time = time.time()
                    agent = rl_planners.PolicyIterationPlanner(env, gamma=gamma, theta=theta)
                elif algorithm == "QL":
                    start_time = time.time()
                    agent = rl_agents.QLearningAgent(env, gamma=gamma, epsilon=epsilon)
                    agent.train()
                elif algorithm == "SARSA":
                    start_time = time.time()
                    agent = rl_agents.SARSAgent(env, gamma=gamma, epsilon=epsilon)
                    agent.train()

                end_time = time.time()

                # Track performance metrics (convergence time and policy stability)
                convergence_time = end_time - start_time

                if algorithm in ["QL", "SARSA","PI"]:
                    stability = np.mean(agent.policy_changes_list)  # Policy stability
                else:
                    stability = np.mean(agent.policy_stability_list)

                results.append((gamma, theta, epsilon, convergence_time, stability))

    return results


def plot_results(algo,hp,results,filename):
    # Unpack results
    gamma_values, theta_values, epsilon_values, convergence_times, stability = zip(*results)

    if hp=="Gamma":
        # Plot convergence time vs. gamma, theta, epsilon
        plt.figure(figsize=(10, 6))
        plt.plot(gamma_values, convergence_times, label="Convergence Time", color='b',marker='o')
        plt.xlabel('Gamma')
        plt.ylabel('Convergence Time (s)')
        plt.title('Convergence Time Sensitivity to Gamma for '+ algo)
        plt.legend()
        plt.grid(True)
        plt.savefig("conv_time_gamma_"+filename)

    elif hp=="Theta":
        plt.figure(figsize=(10, 6))
        plt.plot(theta_values, convergence_times, label="Convergence Time", color='b',marker='o')
        plt.xticks(theta_values)
        plt.xlabel('Theta')
        plt.ylabel('Convergence Time (s)')
        plt.title('Convergence Time Sensitivity to Theta for '+ algo)
        plt.xscale('log')  # Apply log scale
        plt.xticks(theta_values, labels=[f'{x:.0e}' for x in theta_values])  # Format theta as scientific notation

        plt.legend()
        plt.grid(True)
        plt.savefig("conv_time_theta_" + filename)
    elif hp=="Epsilon":
        plt.figure(figsize=(10, 6))
        plt.plot(epsilon_values, convergence_times, label="Convergence Time", color='b',marker='o')
        plt.xlabel('Epsilon')
        plt.ylabel('Convergence Time (s)')
        plt.title('Convergence Time Sensitivity to Epsilon for '+ algo)
        plt.legend()
        plt.grid(True)
        plt.savefig("conv_time_epsilon_" + filename)


#gamma=0.75, alpha=0.1, epsilon=1.0, epsilon_decay=0.99, episodes=2000
#Experiment 1 Convergence time vs Gamma



# Run the sensitivity experiment for Q-Learning (QL), SARSA (SARSA), Policy Iteration (PI), and Value Iteration (VI)

for algo in ["VI","PI","QL","SARSA"]:
    env = TrafficEnv(rewards={"state": 0}, max_steps=1000)  # Create environment

    epsilon_values = [1]
    theta_values = [1e-6]
    gamma_values = [0.1, 0.3, 0.5, 0.75, 0.9]  # Experiment with these values

    results_gamma = run_sensitivity_experiment(env, gamma_values, theta_values, epsilon_values, algorithm=algo)
    plot_results(algo=algo,hp="Gamma",results=results_gamma,filename=algo+"_traffic.png")

    env = TrafficEnv(rewards={"state": 0}, max_steps=1000)


    epsilon_values = [1]
    theta_values = [1e-2,1e-4,1e-6,1e-8]
    gamma_values = [0.9]

    if algo in ["VI","PI"]:
        results_theta = run_sensitivity_experiment(env, gamma_values, theta_values, epsilon_values, algorithm=algo)
        plot_results(algo=algo, hp="Theta", results=results_theta, filename=algo + "_traffic.png")

    env = TrafficEnv(rewards={"state": 0}, max_steps=1000)

    epsilon_values = [0.3,0.5,0.7,0.9]
    theta_values = [1]
    gamma_values = [0.75]

    if algo in ["QL","SARSA"]:
        results_eps = run_sensitivity_experiment(env, gamma_values, theta_values, epsilon_values, algorithm=algo)
        plot_results(algo=algo, hp="Epsilon", results=results_eps, filename=algo + "_traffic.png")


