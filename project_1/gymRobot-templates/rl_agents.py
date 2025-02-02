import numpy as np
import random
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
        # calculate total number of states and initialize Q-table
        self.n_positions = self.env.n_rows * self.env.n_cols
        self.num_workers = len(self.env.warehouse_workers)
        self.total_states = self.calculate_total_states()
        self.q_table = np.zeros((self.total_states, self.env.action_space.n))
        self.noise_scale = 0.1  # For adding noise
        self.temperature = 1  # For softmax
        # initialize cache for indices
        self.index_cache = {}
        self.policy_changes_list = []

    def calculate_total_states(self):
        """
        Calculates the total number of possible states in the environment.

        Returns:
            int: The total number of states in the environment, calculated as:
                 (number of possible robot positions) * (number of bump statuses) * (number of possible target positions).
        """
        return (self.n_positions *  # robot position
                5 *                 # bump status
                self.n_positions)   # current target

    def get_qtable_index(self, encoded_observation):
        """
        Converts the encoded observation into a single index for the Q-table.

        Parameters:
            encoded_observation (np.ndarray): The encoded observation as a 1D numpy array.
                - The first element corresponds to the robot's position in the grid.
                - The second element represents the bump status (no bump, wall, equipment, worker, or target).
                - The third element corresponds to the current target's position in the grid.

        Returns:
            int: The index in the Q-table corresponding to the encoded observation.
        """
        encoded_observation = tuple(encoded_observation)

        # check if the index is already cached
        if encoded_observation in self.index_cache:
            return self.index_cache[encoded_observation]

        # extract components from the encoded observation
        robot_position, bumped_status, curr_target = encoded_observation

        # compute the Q-table index
        qtable_index = (
            int(robot_position) * 5 * self.n_positions +  # robot's position component
            int(bumped_status) * self.n_positions +       # bump status component
            int(curr_target)                              # current target component
        )

        # cache the computed index
        self.index_cache[encoded_observation] = qtable_index

        return qtable_index

    def choose_action(self, state):
        """
        Select an action using epsilon-greedy policy.

        Parameters:
            state (np.ndarray): The current state of the agent in the environment.
        Returns:
            int: The action chosen by the agent.
        """
        state_index = self.get_qtable_index(state)
        if np.random.uniform(0, 1)<self.epsilon:
            action = self.env.action_space.sample()
        else:
            action=np.argmax(self.q_table[state_index])

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
        for episode in range(self.episodes):

            state = self.env.reset()
            done = False
            print("episode", episode)
            policy_changes = 0
            while not done:
                # print("state1")
                # print(state)
                state_index = self.get_qtable_index(state)
                action = self.choose_action(state)
                # print("action")
                # print(action)
                observation, reward, done, truncated, info = self.env.step(action)
                next_state_index = self.get_qtable_index(observation)
                # print("observation")
                # print(observation)
                if done or truncated:
                    self.epsilon *= self.epsilon_decay
                    break

                self.learn(state, action, reward, observation, done)

                if np.argmax(self.q_table[state_index]) != np.argmax(self.q_table[next_state_index]):
                    policy_changes += 1

                state = observation

            self.policy_changes_list.append(policy_changes)
            self.epsilon *= self.epsilon_decay

        end_time = time.time()
        print(f"QL Converged in {end_time - start_time} seconds.")
        plot_policy_stability(self.policy_changes_list, "Robot_Qlearning_policy_stability.png",algo="QL")

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
        state_index = self.get_qtable_index(state)

        # print("state")
        # print(state)
        # print('state_index')
        # print(state_index)
        # print("next_state")
        # print(next_state)
        next_state_index = self.get_qtable_index(next_state)

        max_Q = np.max(self.q_table[next_state_index]) if not done else 0
        self.q_table[state_index][action] += self.alpha * (reward + self.gamma * max_Q - self.q_table[state_index][action])


class SARSAgent(Agent):
    """SARSA agent implementing the on-policy TD control."""

    def train(self):
        """
        Train the agent using the specified RL algorithm.

        Returns:
            np.ndarray: The updated Q-table after training.
        """
        start_time = time.time()
        for episode in range(self.episodes):
            print("episode", episode)
            state = self.env.reset()
            done = False
            policy_changes = 0

            action = self.choose_action(state)
            while not done:
                # print("done", done)
                observation, reward, done, truncated, info = self.env.step(action)
                next_action = self.choose_action(observation)
                self.learn(state, action, reward, observation, done, next_action)

                if action != next_action:
                    policy_changes += 1

                state = observation
                action = next_action

                state_index = self.get_qtable_index(state)
                next_state_index = self.get_qtable_index(observation)

                # if np.argmax(self.q_table[state_index]) != np.argmax(self.q_table[next_state_index]):
                #     policy_changes += 1

                if done or truncated:
                    state_index = state_index = self.get_qtable_index(state)
                    self.q_table[state_index] = 0
                    self.epsilon *= self.epsilon_decay
                    break
            self.policy_changes_list.append(policy_changes)
            self.epsilon *= self.epsilon_decay

        end_time = time.time()

        print(f"SARSA Converged in {end_time - start_time} seconds.")

        plot_policy_stability(self.policy_changes_list, "Robot_SARSA_policy_stability.png", algo="SARSA")

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
        state_index = self.get_qtable_index(state)

        # next_action = self.choose_action(next_state)
        next_state_index = self.get_qtable_index(next_state)

        max_Q = self.q_table[next_state_index][next_action] if not done else 0
        self.q_table[state_index][action] += self.alpha * (reward + self.gamma * max_Q - self.q_table[state_index][action])

def plot_policy_stability(policy_stability_list, filename,algo):
    """
    Plot policy stability across iterations of Value Iteration.
    Tracks the number of times the policy changes.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(policy_stability_list, label="Policy Changes", color='b')
    plt.xlabel('Iteration Number')
    plt.ylabel('Policy Changes')
    plt.title('Policy Stability Across Iterations of '+ algo)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    #plt.show()
