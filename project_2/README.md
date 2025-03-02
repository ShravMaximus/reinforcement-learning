[PROJECT 2]

# Lunar Lander Continuous

## Project Overview

This project involves the implementation of DDPG algorithm to solve the **Lunar lander continous** problem where an agent must land a spacecraft safely on a designated pad by controlling its thrusters.

## Structure of the Repository

[UPDATED] The gymnasium compliant implementations are organized as follows:

``` bash
Lunar-lander.py
```

-   `Lunar-lander.py`: Contains the implementation of the DDPG algorithm and its associated classes `actor_nn` (actor network), `critic_nn` (critic network), `replayBuffer` (for sampling from the replay buffer) along with the functions `noise_process` (for OU noise), `train` for training the model, and code for testing the model and hyperparameter tuning (`hp_tuning`).

## How to Run

1.  Install the libraries in the `requirements.txt` provided in the folder for each problem and then follow the next steps.

2.  To generate the metrics and graphs with the finalized hyperparameters execute all the code **before** the comment `##HYPER PARAMETER TUNING FUNCTION`. This will generate `training_rewards_plot.png` (Reward at each training episode), `test_rewards_plot.png` (Reward per episode for 100 consecutive episodes using the trained agent) and `ddpg_test_results.csv` - a .csv file containing the test rewards are each of the 100 episode. The `Reward` column should be averaged to get the average reward over 100 consecutive runs. Additionally, the actor and critic neural networks are saved as `ddpg_actor.pth` and `ddpg_critic.pth` respectively.

3.  To tune the hyper parameters execute the code **after** the comment `##HYPER PARAMETER TUNING FUNCTION`. Either the code for each hyperparameter can be executed seprately or all together. The code will generate .csv file with the following naming convention - `ddpg_test_results_`+`<hyperparameter_name>`+`_`+`<hyperparameter_value>`. For example: `ddpg_test_results_gamma_0.9.csv`, `ddpg_test_results_lr_0.0001.csv`, `ddpg_test_results_tau_0.0025.csv` etc. Each file will contain the rewards for consecutive 100 episodes and the `Reward` column should be averaged to get the average reward over 100 consecutive runs.

4.  Alternatively, running the complete `Lunar-lander.py` file will generate all mentioned files but requires significant processing time.
