[STUDENTS: MAKE SURE YOU EDIT THIS README WITH ADDITIONAL INFORMATION RELATING TO EXPERIMENTS AND OTHER FILES YOU CREATE]

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
├── rl_planners.py
└── rl_agents.py
```
- `traffic_simulator.py`: Contains the classes `TrafficSim` to simulate the arrival and departure of cars from the intersection and `TrafficRenderer` that creates a visual representation of the intersection. 
- `traffic_environment.py`: Contains the class `TrafficEnv` that implements the MDP and interfaces with simulation to step and update the state of the world.
- `traffic_execution.py`: Sample execution of an episode of the `TrafficEnv` MDP with actions chosen by an agent or planner.
- `rl_planners.py`: Contains implementations of planners containing including value iteration and policy iteration.
- ` rl_agents.py`: Contains implementations of SARSA and Q-Learning to obtain agents for the traffic problem.

```bash
gymRobot-templates/
├── robot_simulator.py
├── robot_environment.py
├── robot_execution.py
└── rl_agents.py
```
- `robot_simulator.py`: Contains the classes `RobotSim` to simulate the robot moving about the factory and delivering boxes and `RobotRenderer` that creates a visual representation of the gridworld of the warehouse.
- `robot_environment.py`: Contains the class `RobotEnv` that implements the MDP and interfaces with simulation to step and update the state of the world.
- `robot_execution.py`: Sample execution of an episode of the `RobotEnv` MDP with actions chosen by an agent or planner.
- ` rl_agents.py`: Contains implementations of SARSA and Q-Learning to obtain agents for the traffic problem.
