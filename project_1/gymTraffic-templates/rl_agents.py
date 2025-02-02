import numpy as np
import time
import matplotlib.pyplot as plt

class Agent:
    """
    Base RL agent class for Q-Learning and SARSA.

    Args:
        env (gym.Env): The environment to train in.
        gamma (float): Discount factor for future rewards.
        alpha (float): Learning rate.
        epsilon (float): Exploration rate.
        epsilon_decay (float): Decay factor for epsilon.
        episodes (int): Number of training episodes.
    """

    def __init__(self, env, gamma=0.75, alpha=0.1, epsilon=1.0, epsilon_decay=0.99, episodes=2000):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        #self.q_table = np.zeros((*self.env.observation_space.nvec, self.env.action_space.n))

        sizes = [self.env.max_cars_dir + 1, self.env.max_cars_dir + 1, 2]
        self.all_states = np.indices(sizes).reshape(len(sizes), -1).T
        self.state_to_index = {tuple(state): index for index, state in enumerate(self.all_states)}
        self.q_table = np.zeros((len(self.all_states), self.env.action_space.n), dtype=np.float64)
        self.policy_changes_list = []


    def choose_action(self, state):
        """
        Select an action using epsilon-greedy policy.

        Returns:
            int: The action chosen by the agent.
        """

        state_index = self.state_to_index[tuple(state)]

        if np.random.uniform(0,1) < self.epsilon:
            action = self.env.action_space.sample() #chose randin action
        else:
            action=np.argmax(self.q_table[state_index]) #choose best action with max Q value

        return action

class QLearningAgent(Agent):
    """Q-Learning agent implementing the off-policy TD control."""

    def train(self):
        """
        Train the agent using the specified RL algorithm.

        Returns:
            np.ndarray: The updated Q-table after training.
        """
        start_time = time.time()
        delta_list = []
        for episode in range(self.episodes):
            tr_cumm_reward = 0
            state = self.env.reset()
            done = False
            #print("episode", episode)
            policy_changes = 0
            previous_q_table = self.q_table.copy()
            previous_action = self.choose_action(state)
            while not done:
                # print("state1")
                # print(state)
                action = self.choose_action(state)
                # print("action")
                # print(action)
                observation, reward, done, truncated, info = self.env.step(action)
                tr_cumm_reward+=reward
                # print("observation")
                # print(observation)
                if done or truncated:
                    self.epsilon *= self.epsilon_decay
                    break
                self.learn(state, action, reward, observation, done)

                delta = np.max(np.abs(self.q_table - previous_q_table))
                delta_list.append(delta)
                previous_q_table = self.q_table.copy()

                state_index = self.state_to_index[tuple(state)]
                next_state = np.array(observation)
                next_state_index = self.state_to_index[tuple(next_state)]

                if np.argmax(self.q_table[state_index]) != np.argmax(self.q_table[next_state_index]):
                    policy_changes += 1

                state = observation

            self.policy_changes_list.append(policy_changes)
            #print("episode:", episode,"tr_cumm_reward:",tr_cumm_reward)
            self.epsilon *= self.epsilon_decay

        end_time = time.time()
        print(f"QL Converged in {end_time - start_time} seconds.")
        plot_policy_stability(self.policy_changes_list, "Qlearning_policy_stability.png",algo="QL")
        plot_convergence_QL(delta_list, "Qlearning_convergence.png")

        return self.q_table

    def learn(self, state, action, reward, next_state, done):
        """
        Update the Q-table using the Q-Learning algorithm.

        Args:
            state (tuple): The current state of the environment.
            action (int): The action taken by the agent.
            reward (float): The reward received after taking the action.
            next_state (tuple): The next state of the environment.
            done (bool): Whether the episode is complete.

        Returns:
            None
        """
        state_index = self.state_to_index[tuple(state)]



        # print("state")
        # print(state)
        # print('state_index')
        # print(state_index)
        # print("next_state")
        next_state = np.array(next_state)
        #print(next_state)
        next_state_index = self.state_to_index[tuple(next_state)]

        max_Q=np.max(self.q_table[next_state_index]) if not done else 0
        self.q_table[state_index][action]+= self.alpha*(reward + self.gamma*max_Q - self.q_table[state_index][action])


class SARSAgent(Agent):
    """SARSA agent implementing the on-policy TD control."""

    def train(self):
        """
        Train the agent using the specified RL algorithm.

        Returns:
            np.ndarray: The updated Q-table after training.
        """
        start_time = time.time()
        delta_list = []
        for episode in range(self.episodes):
            #print("episode",episode)
            state = self.env.reset()
            done = False
            action=self.choose_action(state)
            policy_changes = 0
            previous_q_table = self.q_table.copy()

            while not done:
                #print("done", done)
                observation, reward, done, truncated, info = self.env.step(action)
                next_action=self.choose_action(observation)
                self.learn(state, action, reward, observation, done,next_action)
                delta = np.max(np.abs(self.q_table - previous_q_table))  # Delta: max change in Q-table
                delta_list.append(delta)
                previous_q_table = self.q_table.copy()  # upd previous Q-table for the next step

                state_index = self.state_to_index[tuple(state)]
                next_state = np.array(observation)
                next_state_index = self.state_to_index[tuple(next_state)]

                if np.argmax(self.q_table[state_index]) != np.argmax(self.q_table[next_state_index]):
                    policy_changes += 1

                state = observation
                action = next_action
                if done or truncated:
                    state_index = self.state_to_index[tuple(state)]
                    self.q_table[state_index] = 0
                    self.epsilon *= self.epsilon_decay
                    break
            self.policy_changes_list.append(policy_changes)
            self.epsilon *= self.epsilon_decay

        end_time = time.time()
        print(f"SARSA Converged in {end_time - start_time} seconds.")
        plot_convergence_SARSA(delta_list, "SARSA_convergence.png")
        plot_policy_stability(self.policy_changes_list, "SARSA_policy_stability.png",algo="SARSA")

        return self.q_table

    def learn(self, state, action, reward, next_state, done,next_action):
        """
        Update the Q-table using the SARSA algorithm.

        Args:
            state (tuple): The current state of the environment.
            action (int): The action taken by the agent.
            reward (float): The reward received after taking the action.
            next_state (tuple): The next state of the environment.
            done (bool): Whether the episode is complete.

        Returns:
            None
        """
        state_index = self.state_to_index[tuple(state)]
        next_state = np.array(next_state)
        #next_action = self.choose_action(next_state)
        next_state_index = self.state_to_index[tuple(next_state)]

        max_Q = self.q_table[next_state_index][next_action] if not done else 0
        self.q_table[state_index][action] += self.alpha * (reward + self.gamma * max_Q - self.q_table[state_index][action])



def plot_policy_stability(policy_stability_list, filename,algo):

    plt.figure(figsize=(10, 6))
    plt.plot(policy_stability_list, label="Policy Changes", color='b')
    plt.xlabel('Iteration Number')
    plt.ylabel('Policy Changes')
    plt.title('Policy Stability Across Iterations of '+ algo)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    #plt.show()

def plot_convergence_QL(delta_list, filename):

    plt.figure(figsize=(10, 6))
    plt.plot(delta_list, label="Delta (Max Q-value Change)", color='g')
    plt.xlabel('Episode Number')
    plt.ylabel('Delta (Max Q-value Change)')
    plt.title('Convergence of Q-Learning')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    #plt.show()

def plot_convergence_SARSA(delta_list, filename):

    plt.figure(figsize=(10, 6))
    plt.plot(delta_list, label="Delta (Max Q-value Change)", color='r')
    plt.xlabel('Episode Number')
    plt.ylabel('Delta (Max Q-value Change)')
    plt.title('Convergence of SARSA')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    #plt.show()
