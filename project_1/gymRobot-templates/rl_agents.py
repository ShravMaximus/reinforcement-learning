import numpy as np

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

        # initialize cache for indices
        self.index_cache = {}

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
        pass


class QLearningAgent(Agent):
    """Q-Learning agent implementing the off-policy TD control."""

    def train(self):
        """
        Train the agent using the specified RL algorithm.

        Returns:
            np.ndarray: The updated Q-table after training.
        """
        pass

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
        pass


class SARSAgent(Agent):
    """SARSA agent implementing the on-policy TD control."""

    def train(self):
        """
        Train the agent using the specified RL algorithm.

        Returns:
            np.ndarray: The updated Q-table after training.
        """
        pass

    def learn(self, state, action, reward, next_state, done):
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
        pass
