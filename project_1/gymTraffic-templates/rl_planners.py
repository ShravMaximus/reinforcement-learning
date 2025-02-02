import numpy as np
import time
import matplotlib.pyplot as plt

class ValueIterationPlanner:
    """
    Planner that uses the Value Iteration algorithm to determine the optimal policy for a given environment.

    Args:
        env (gym.Env): The traffic environment.
        gamma (float): Discount factor for future rewards.
        theta (float): Threshold for stopping value iteration (determines convergence).
    """
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        sizes = [self.env.max_cars_dir + 1, self.env.max_cars_dir + 1, 2]
        self.all_states = np.indices(sizes).reshape(len(sizes), -1).T
        self.state_to_index = {tuple(state): index for index, state in enumerate(self.all_states)}
        self.policy = np.random.choice([0, 1], size=np.array(sizes).prod())  # initialize policy arbitrarily
        self.value_function = np.zeros(np.array(sizes).prod())
        self.policy_stability_list = []  # To track policy changes across iterations

        self.value_iteration()

    def value_iteration(self):
        """
        Perform value iteration to compute the optimal value function and policy.
        """
        num_iterations = 0
        delta_list = []
        start_time = time.time()
        while True:
            delta = 0
            num_iterations+=1
            Q = np.zeros((len(self.all_states), self.env.action_space.n), dtype=np.float64)
            policy_changes = 0
            previous_policy = self.policy.copy()
            previous_best_actions = np.copy(self.policy)
            # iterate over all states
            for state_index, state_array in enumerate(self.all_states):
                state = tuple(state_array)
                # iterate over all actions
                for a in range(self.env.action_space.n):
                    # calculate the expected value of taking action `a` in state `state`
                    for prob, next_state, reward, done in self.env.P[state][a]:
                        next_state_index = self.state_to_index[tuple(next_state)]
                        Q[state_index, a] += prob * (reward + self.gamma * self.value_function[next_state_index] * (not done))
                # update the value function with the best action value
                best_action_value = np.max(Q[state_index])
                delta = max(delta, np.abs(self.value_function[state_index] - best_action_value))
                self.value_function[state_index] = best_action_value

            for state_index in range(len(self.all_states)):
                best_action = np.argmax(Q[state_index])
                self.policy[state_index] = best_action

            #print("previous_best_actions",previous_best_actions)
            #print("policy",self.policy)
            policy_changes = np.sum(previous_best_actions != self.policy)  # Compare the best actions
            #print("policy_changes",policy_changes)
            self.policy_stability_list.append(policy_changes)
            delta_list.append(delta)
            # check for convergence
            if delta < self.theta:
                break

        end_time = time.time()
        print(f"Value Iteration Converged in {end_time - start_time} seconds.")

        plot_convergence_VI(delta_list, "VI_traffic_convergence_plot.png")
        plot_policy_stability_VI(self.policy_stability_list, "VI_policy_stability_plot.png")

        # extract the policy from the Q-table
        for state_index in range(len(self.all_states)):
            self.policy[state_index] = np.argmax(Q[state_index])

    def choose_action(self, state):
        """
        Select the action based on the learned policy.

        Args:
            state (tuple): The current state of the environment.

        Returns:
            int: The action chosen by the policy.
        """
        state_index = self.state_to_index[tuple(state)]
        return self.policy[state_index]


class PolicyIterationPlanner:
    """
    Planner that uses the Policy Iteration algorithm to determine the optimal policy for a given environment.

    Args:
        env (gym.Env): The traffic environment.
        gamma (float): Discount factor for future rewards.
        theta (float): Threshold for stopping policy evaluation.
    """

    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        sizes = [self.env.max_cars_dir + 1, self.env.max_cars_dir + 1, 2]
        self.all_states = np.indices(sizes).reshape(len(sizes), -1).T
        self.state_to_index = {tuple(state): index for index, state in enumerate(self.all_states)}
        self.policy = np.random.choice([0, 1], size=np.array(sizes).prod())  # initialize policy arbitrarily
        self.value_function = np.zeros(np.array(sizes).prod())
        self.policy_changes_list=[]
        self.policy_iteration()

    def evaluate_policy(self):
        """
        Evaluate the current policy by computing the value function for all states.

        Returns:
            None
        """
        delta_list = []
        while True:
            delta = 0
            # iterate over all states
            for state_index, state_array in enumerate(self.all_states):
                state = tuple(state_array)
                a = self.policy[state_index]
                value=0
                # calculate the expected value of taking action `a` in state `state`
                for prob, next_state, reward, done in self.env.P[state][a]:
                    next_state_index = self.state_to_index[tuple(next_state)]
                    value += prob * (reward + self.gamma * self.value_function[next_state_index] * (not done))

                delta = max(delta, np.abs(self.value_function[state_index] - value))
                self.value_function[state_index] = value
            delta_list.append(delta)
            # check for convergence
            if delta < self.theta:
                #print("policy eval converge")
                #time.sleep(10)
                break
        return delta_list


    def improve_policy(self):
        """
        Improve the current policy by making it greedy with respect to the current value function.

        Returns:
            bool: True if the policy is stable (i.e., no changes were made), False otherwise.
        """
        stable_policy=True
        policy_changes = 0 #policy changes tracking
        previous_policy = self.policy.copy()
        for state_index, state_array in enumerate(self.all_states):
            state = tuple(state_array)
            a_best = self.policy[state_index]
            max_value=-float('inf')
            for a in range(self.env.action_space.n):
                value=0
                for prob, next_state, reward, done in self.env.P[state][a]:
                    next_state_index = self.state_to_index[tuple(next_state)]
                    value += prob * (reward + self.gamma * self.value_function[next_state_index] * (not done))

                if value > max_value:
                    max_value=value
                    a_best = a

            if self.policy[state_index] != a_best:
                stable_policy=False
                self.policy[state_index] = a_best
                policy_changes += 1

        #print("policy_changes",policy_changes)
        #policy_changes = np.sum(previous_policy != self.policy)
        self.policy_changes_list.append(policy_changes)
        return stable_policy


    def policy_iteration(self):
        """
        Perform policy iteration by alternately evaluating and improving the policy.

        Returns:
            np.ndarray: The final policy after convergence.
        """
        num_iterations = 0
        delta_all_iterations = []
        policy_stability_list = []
        start_time = time.time()
        while True:
            delta_list = self.evaluate_policy()
            delta_all_iterations.append(delta_list)

            self.evaluate_policy()
            stable_policy=self.improve_policy()
            num_iterations+=1
            if stable_policy==True:
                print(f"policy iter converged after {num_iterations} iterations")
                #time.sleep(10)
                break
        end_time = time.time()
        print(f"Policy Iteration Converged in {end_time - start_time} seconds.")

        plot_policy_stability(self.policy_changes_list, "PI_policy_stability_plot.png")

        plot_convergence(delta_all_iterations,"PI_traffic_convergence_plot.png")


    def choose_action(self, state):
        """
        Select the action based on the learned policy.

        Args:
            state (tuple): The current state of the environment.

        Returns:
            int: The action chosen by the policy.
        """
        idx = self.state_to_index[tuple(state)]
        return self.policy[idx]

def plot_convergence(delta_all_iterations, filename):

    plt.figure(figsize=(10, 6))

    for i, delta_list in enumerate(delta_all_iterations):
        plt.plot(delta_list, label=f'Iteration {i + 1}')

    plt.xlabel('Iteration Number')
    plt.ylabel('Delta (Max Value Function Change)')
    plt.title('Convergence of Policy Iteration')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    #plt.show()

def plot_policy_stability(policy_changes, filename):

    plt.figure(figsize=(10, 6))

    plt.plot(range(len(policy_changes)), policy_changes, label='Policy Changes', color='blue')

    plt.xlabel('Iteration Number')
    plt.ylabel('Policy Changes')
    plt.title('Policy Stability Across Iterations of PI')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)

def plot_policy_stability_VI(policy_stability_list, filename):

    plt.figure(figsize=(10, 6))
    plt.plot(policy_stability_list, label="Policy Changes", color='b')
    plt.xlabel('Iteration Number')
    plt.ylabel('Policy Changes')
    plt.title('Policy Stability Across Iterations of VI')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    #plt.show()

def plot_convergence_VI(delta_list, filename):

    plt.figure(figsize=(10, 6))
    plt.plot(delta_list, label="Delta (Max Value Function Change)", color='b')
    plt.xlabel('Iteration Number')
    plt.ylabel('Delta (Max Value Function Change)')
    plt.title('Convergence of Value Iteration')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    #plt.show()