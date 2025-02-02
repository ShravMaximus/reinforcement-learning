[PROJECT 1]

# Reinforcement learning for Traffic Control and Warehouse Robotics

## Project Overview

This project involves the implementation of reinforcement learning algorithms to solve two distinct problems:
1. **Traffic Control at an intersection**: Optimizing traffic light control to ensure efficient traffic flow.
2. **Warehouse robot navigation**: Enabling a robot to autonomously navigate a warehouse, collecting items and avoiding obstacles.

## Structure of the Repository

[UPDATED]
The gymnasium compliant implementations are organized as follows:
```bash
gymTraffic-templates/
├── traffic_simulator.py
├── traffic_environment.py
├── traffic_execution.py
├── traffic_experiments.py
├── rl_planners.py
├── requirements.txt
└── rl_agents.py
```
- `traffic_simulator.py`: Contains the classes `TrafficSim` to simulate the arrival and departure of cars from the intersection and `TrafficRenderer` that creates a visual representation of the intersection. 
- `traffic_environment.py`: Contains the class `TrafficEnv` that implements the MDP and interfaces with simulation to step and update the state of the world.
- `traffic_execution.py`: Sample execution of an episode of the `TrafficEnv` MDP with actions chosen by an agent or planner.
- ` traffic_experiments.py`: Contains implementations and graphs of the sensitivity analysis for VI,PI, SARSA Q-Learning for the traffic problem.
- `rl_planners.py`: Contains implementations of planners containing including value iteration and policy iteration.
- ` requirements.txt`: Contains the required libraries needed to execute the files
- ` rl_agents.py`: Contains implementations of SARSA and Q-Learning to obtain agents for the traffic problem.

```bash
gymRobot-templates/
├── robot_simulator.py
├── robot_environment.py
├── robot_execution.py
├── robot_experiments.py
├── requirements.txt
└── rl_agents.py
```
- `robot_simulator.py`: Contains the classes `RobotSim` to simulate the robot moving about the factory and delivering boxes and `RobotRenderer` that creates a visual representation of the gridworld of the warehouse.
- `robot_environment.py`: Contains the class `RobotEnv` that implements the MDP and interfaces with simulation to step and update the state of the world.
- `robot_execution.py`: Sample execution of an episode of the `RobotEnv` MDP with actions chosen by an agent or planner.
- ` robot_experiments.py`: Contains implementations and graphs of the sensitivity analysis for SARSA Q-Learning for the warehouse problem.
- ` requirements.txt`: Contains the required libraries needed to execute the files
- ` rl_agents.py`: Contains implementations of SARSA and Q-Learning to obtain agents for the warehouse problem.

## How to Run
1. Install the libraries in the `requirements.txt` provided in the folder for each problem and then follow the next steps.
2. To generate the metrics, policty stability and convergence plots, run the `traffic_execution.py` (for traffic problem) and `robot_execution.py` (for the warehouse problem). These files will generate the policty stability and convergence plots, along with the .csv metrics file for each algorithm and save in the working directory. The time to convergence metric will be printed on the screen and can be obtained from there.
3. To generate the graphs for the sensitivity analysis, `traffic_experiments.py` (for traffic problem) and `robot_experiments.py` (for the warehouse problem) should be executed and all the graphs will be generated in the working directory.
4. The reward structure is fixed according to the problem, but can be modified as explained in the reward to generate metrics for other structures.

