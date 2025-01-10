import sys
import numpy as np
from typing import Optional
import gymnasium as gym
from gymnasium import spaces

from robot_simulator import RobotSim
from robot_simulator import RobotRenderer


class RobotEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, n_rows=10, n_cols=10, warehouse_workers=None, warehouse_equipment=None, start = None, goal= None, rewards=None, prob_success=1.0, max_steps=1000, renderer=None):
        """
        Initializes the RobotEnv environment.

        Parameters:
            n_rows (int, optional): Number of rows in the warehouse grid. Default is 10.
            n_cols (int, optional): Number of columns in the warehouse grid. Default is 10.
            warehouse_workers (list of tuples, optional): Initial positions of the warehouse workers. Default is [(2, 6), (7, 6)].
            warehouse_equipment (list of tuples, optional): Positions of the warehouse equipment. Default is [(4, 2), (4, 3), (4, 4), (4, 5)].
            start (tuple, optional): Starting position of the robot. Default is randomly sampled.
            goal (tuple, optional): The goal position where the robot should deliver boxes. Default is randomly sampled.
            rewards (dict): Dictionary defining the reward values for different events. Default is None.
            prob_success (float, optional): Probability of successfully executing the chosen action. Default is 1.0.
            max_steps (int, optional): Maximum number of steps allowed per episode. Default is 10000.
            renderer (RobotRenderer, optional): The renderer used for visualizing the environment. Default is None.
        """
        # set the warehouse layout
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.warehouse_workers = warehouse_workers or [(2, 6), (7, 6)]
        self.warehouse_equipment = warehouse_equipment or [(4, 2), (4, 3), (4, 4), (4, 5)]
        # sample start and goal positions if not provided
        all_positions = [(r, c) for r in range(n_rows) for c in range(n_cols)]
        occupied_positions = set(self.warehouse_workers + self.warehouse_equipment)
        valid_positions = [pos for pos in all_positions if pos not in occupied_positions]
        if len(valid_positions) < 2:
            raise ValueError("Not enough valid positions for start and goal.")
        self.start = start or random.choice(valid_positions)
        valid_positions.remove(self.start)
        self.goal = goal or random.choice(valid_positions)
        # set the rewards structure
        self.rewards = rewards
        # setting max number of steps per episode and tracking steps
        self.max_steps = max_steps
        self.current_step = 0
        # define action and observation spaces
        self.action_space = spaces.Discrete(4) # up, right, left, down
        # total number of positions in the grid
        n_positions = n_rows * n_cols
        # define the observation space
        self.observation_space = spaces.Dict({
            "robot_position": spaces.Discrete(n_positions),
            "bumped_status": spaces.Discrete(5),  # 5 possible bump statuses ("", "wall", "equipment", "worker", "target")
            "curr_target": spaces.Discrete(n_positions),
        })
        # define transition probabilities
        self.prob_success = prob_success
        self.prob_fail = (1.0 - self.prob_success) / (self.action_space.n - 1)

        # initialize simulator with environment parameters
        self.sim = RobotSim(n_rows, n_cols, self.warehouse_workers, self.warehouse_equipment, self.goal, self.start)

        # initialize the state
        self.s = self.sim.get_world_state()

        # initialize renderer
        self.renderer = None

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        """
        Resets the environment to the initial state.

        Parameters:
            seed (int, optional): Random seed for reproducibility. Default is None.
            return_info (bool, optional): Whether to return additional information. Default is False.
            options (dict, optional): Additional options for resetting the environment. Default is None.

        Returns:
            np.ndarray: The initial observation representing the robot's starting position.
            dict (optional): Additional information if return_info is True.
        """
        # reset the simulation state
        self.sim.reset(self.n_rows, self.n_cols, self.warehouse_workers, self.warehouse_equipment, self.goal, self.start)
        self.current_step = 0
        self.renderer = RobotRenderer(self.sim)
        self.s = self.sim.get_world_state()

        if return_info:
            return self.encode_observation(self.s), {}
        return self.encode_observation(self.s)

    def sample_action(self, action):
        """
        Sample an action based on the given action and the environment's transition probabilities.

        Parameters:
            action (int): The action chosen by the agent, where 0 = up, 1 = right, 2 = down, and 3 = left.

        Returns:
            tuple: A tuple containing the probability and the action actually taken.
        """
        pass

    def step(self, action):
        """
        Takes a step in the environment based on the action.

        Parameters:
            action (int): The action to take, where 0 = up, 1 = right, 2 = down, and 3 = left.

        Returns:
            np.ndarray: The new observation after taking the action.
            float: The reward obtained after taking the action.
            bool: Whether the episode has ended due to a terminal state.
            bool: Whether the episode has ended due to truncation (max steps reached).
            dict: Additional information such as the probability of the action being taken.
        """
        pass

    def get_rewards(self, bumped):
        """
        Calculates the reward for the current state.

        Parameters:
            bumped (str): The object the robot bumped into ("wall", "equipment", "worker", "target", or "").

        Returns:
            float: The reward based on the current state and the object bumped into.
        """
        pass

    def is_truncated(self):
        """
        Checks if the episode has reached the maximum number of steps.

        Returns:
            bool: True if the maximum number of steps has been reached, False otherwise.
        """
        pass

    def is_terminal(self, bumped):
        """
        Checks if the current state is terminal.

        Parameters:
            bumped (str): The object the robot bumped into ("wall", "equipment", "worker", "target", or "").

        Returns:
            bool: True if the state is terminal, False otherwise.
        """
        pass


    def render(self, close=False):
        """
        Render the environment.
        """
        if close and self.renderer:
            if self.renderer:
                self.renderer.close()
            return

        if self.renderer:
            return self.renderer.render()

    def encode_observation(self, world_state):
        """
        Encodes the given world state using linear indices for compactness.

        Parameters:
            world_state (tuple): The state of the world as returned by the get_world_state method.

        Returns:
            np.ndarray: The encoded state as a 1D numpy array.
        """
        # unpack the world state tuple
        row, col, bumped, n_boxes, prev_target, curr_target, workers_tuplist = world_state
        # encode the robot's position as a linear index
        robot_position = np.array([row * self.n_cols + col])
        # encode the bumped status
        bump_mapping = {"": 0, "wall": 1, "equipment": 2, "worker": 3, "target": 4}
        bumped_encoded = np.array([bump_mapping[bumped]])
        # encode the current target as a linear index
        curr_target_encoded = np.array([curr_target[0] * self.n_cols + curr_target[1]])
        # combine all encodings into a single vector
        encoded_state = np.concatenate([
            robot_position,
            bumped_encoded,
            curr_target_encoded,
        ])

        return encoded_state
