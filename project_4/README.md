[PROJECT 4]

# AWS Deepracer

## Project Overview

The objective in this project is to train an autonomous racing agent to successfully navigate a simulated DeepRacer vehicle across a variety of tracks- A to Z Speedway (reInvent2019 wide), Smile Speedway (reInvent2019 track), and Vegas (Vegas_track) and race types (Time-trial, Obstacle-avoidance, and Head-to-bot racing)

## Structure of the Repository

[UPDATED] The main files of the implementations are organized as follows:

P4_deepracer/src/
├── agents.py
├── run.py

P4_deepracer/demos/
├── reInvent2019_wide-head_to_bot-ppo_agent-episode-0.mp4
├── reInvent2019_wide-obstacle_avoidance-ppo_agent-episode-0.mp4
├── reInvent2019_wide-time_trial-ppo_agent-episode-0.mp4


P4_deepracer/evaluations/
├── head_to_bot-ppo_agent.json
├── obstacle_avoidance-ppo_agent.json
├── time_trial-ppo_agent.json

P4_deepracer/plots/
├── *.png

P4_deepracer/
├── requirements.txt
├── test.py
├── visual.py
├── training_logs_part1_tt.txt
├── training_logs_part2_bots.txt
├── training_logs_part2_obs.txt

Demo Videos

[Watch Time-Trial Demo](demos/reInvent2019_wide-time_trial-ppo_agent-episode-0.mp4)
[Watch Head-to-Bot Demo](demos/reInvent2019_wide-head_to_bot-ppo_agent-episode-0.mp4)
[Watch Obstacle-avoidance Demo](demos/reInvent2019_wide-obstacle_avoidance-ppo_agent-episode-0.mp4)


```
- `agents.py`: Contains the implementation of the PPO Algorithm 
- `run.py`: Contains the code to perform the training and save the interim (based on progress thresholds) and final model
- `test.py`: Contains the code to call the training, metrics and demo functions
- `visual.py`: Contains the code to geerate the charts for training and evaluation including the box plots
- `P4_deepracer/demos/`: Contains the demo videos
- `requirements.txt`: Contains the required libraries needed to execute the files
- `P4_deepracer/evaluations/`: Contains evaluation results in .json files

## How to Run

1.  Install the libraries in the `requirements.txt` provided in the folder and then follow the next steps.`
2.  To prepare for training in run.py provide the name of the log file ("log_txt_path") and the name of the model to be saved; you can also use the default values but note them.

2.  To train the models call the run function using the sample code provided in test.py file. The custom params like number of episodes (num_episodes)  and loading weights of a pretrained model for transfer learning are set here.

3. Once model is trained usingn the sample provided in test.py, call evaluate and demo functions in the src.utils. The evaluate function will generate the .json files in P4_deepracer/evaluations/ and the demo function will generate the demo in P4_deepracer/demos/

4. To generate the plots, use the log file thats generated post training and the .json files post evaluate call to update the visual.py file (see example) and this will generate the training plots and the evaluation box plots

5. Note: Start the Obstacle-Avoidance and Head-to-Bot race types by setting NUMBER OF OBSTACLES as 6 and NUMBER OF BOT CARS as 3 respectively one at a time. Keep these both as 0 for Time-trial.
