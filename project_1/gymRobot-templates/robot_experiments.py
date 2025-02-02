import os
#os.chdir('gymRobot-templates')
import time
import gymnasium as gym
from robot_environment import RobotEnv
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
        for alpha in alpha_values:
            for epsilon in epsilon_values:
                # Initialize the agent with the given parameters
                if algorithm == "QL":
                    start_time = time.time()
                    agent = rl_agents.QLearningAgent(env, gamma=gamma, epsilon=epsilon,alpha=alpha)
                    agent.train()
                elif algorithm == "SARSA":
                    start_time = time.time()
                    agent = rl_agents.SARSAgent(env, gamma=gamma, epsilon=epsilon,alpha=alpha)
                    agent.train()

                end_time = time.time()

                # Track performance metrics (convergence time and policy stability)
                convergence_time = end_time - start_time

                if algorithm in ["QL", "SARSA","PI"]:
                    stability = np.mean(agent.policy_changes_list)  # Policy stability
                else:
                    stability = np.mean(agent.policy_stability_list)

                results.append((gamma, alpha, epsilon, convergence_time, stability))

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

    elif hp=="Alpha":
        plt.figure(figsize=(10, 6))
        plt.plot(alpha_values, convergence_times, label="Convergence Time", color='b',marker='o')
        plt.xlabel('Alpha')
        plt.ylabel('Convergence Time (s)')
        plt.title('Convergence Time Sensitivity to Alpha for '+ algo)
        plt.legend()
        plt.grid(True)
        plt.savefig("conv_time_alpha_" + filename)
    elif hp=="Epsilon":
        plt.figure(figsize=(10, 6))
        plt.plot(epsilon_values, convergence_times, label="Convergence Time", color='b',marker='o')
        plt.xlabel('Epsilon')
        plt.ylabel('Convergence Time (s)')
        plt.title('Convergence Time Sensitivity to Epsilon for '+ algo)
        plt.legend()
        plt.grid(True)
        plt.savefig("conv_time_epsilon_" + filename)

# Run the sensitivity experiment for Q-Learning (QL), SARSA (SARSA)

for algo in ["QL","SARSA"]:
    env = RobotEnv(rewards={"state": 0}, max_steps=1000)  # Create environment

    epsilon_values = [1]
    alpha_values = [0.1]
    gamma_values = [0.1, 0.3, 0.5, 0.75, 0.9]  # Experiment with these values

    results_gamma = run_sensitivity_experiment(env, gamma_values, alpha_values, epsilon_values, algorithm=algo)
    plot_results(algo=algo,hp="Gamma",results=results_gamma,filename=algo+"_robot.png")

    env = RobotEnv(rewards={"state": 0}, max_steps=1000)


    epsilon_values = [1]
    alpha_values = [0.1,0.5,0.7,0.9]
    gamma_values = [0.75]

    if algo in ["QL","SARSA"]:
        results_alpha = run_sensitivity_experiment(env, gamma_values, alpha_values, epsilon_values, algorithm=algo)
        plot_results(algo=algo, hp="Alpha", results=results_alpha, filename=algo + "_robot.png")

    env = RobotEnv(rewards={"state": 0}, max_steps=1000)

    epsilon_values = [0.3,0.5,0.7,0.9]
    alpha_values = [0.1]
    gamma_values = [0.75]

    if algo in ["QL","SARSA"]:
        results_eps = run_sensitivity_experiment(env, gamma_values, alpha_values, epsilon_values, algorithm=algo)
        plot_results(algo=algo, hp="Epsilon", results=results_eps, filename=algo + "_robot.png")


