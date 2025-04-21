from src.utils import make_environment
from src.agents import PPOAgent
import numpy as np
import torch
import random

seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


env = make_environment("deepracer-v0")


agent = PPOAgent(env.observation_space, env.action_space)
num_episodes = 1500
episode_rewards = []


for episode in range(num_episodes):

    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0

    while not done:
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)

        action, logp, val = agent.get_action(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 4. Store experience
        agent.store(obs, action, reward, val, logp)
        total_reward += reward
        obs = next_obs

    agent.finish_trajectory(last_val=0.0)
    agent.update()
    # Log reward
    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}/{num_episodes} - Reward: {total_reward:.2f}")

env.close()


from src.run import run

#  custom parameters 
custom_params = {
    'num_episodes': 2000,
    'experiment_name': 'my_custom_run'
}
run(custom_params)

import torch

# loa the agent 
agent_test = torch.load('ppo_agent_progress_100_13_bots.torch', weights_only=False) 
agent_eval=agent_test.eval()  

# eval + demo
from src.utils import evaluate, demo
metrics = evaluate(agent_eval)
demo(agent_eval)

from src.utils import plot_metrics

plot_metrics(
    metrics, title='usage'
)



#obstavcles train
import torch
agent = torch.load("ppo_agent2000.torch", weights_only=False)
torch.save(agent.model.state_dict(), "ppo_agent2000.pth")

#obstavcles run

from src.run import run
custom_params = {
    'num_episodes': 1496,
    'experiment_name': 'my_custom_run',
    'pretrained_model_path': '/home/ubuntu/P4_deepracer/ppo_agent2000.pth'  
}
run(custom_params)


