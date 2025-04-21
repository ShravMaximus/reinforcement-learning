import yaml
import time
import torch
import datetime
import numpy as np
from loguru import logger
from munch import munchify
from torch.utils.tensorboard import SummaryWriter

from src.agents import RandomAgent
from src.agents import PPOAgent
from src.utils import (
    device,
    set_seed,
    make_environment,
)
from src.utils import lap_time
import os

episode_logs = []
log_txt_path = "training_logs_part2_bots.txt"

DEVICE = device()
HYPER_PARAMS_PATH: str = 'configs/hyper_params.yaml'


def tensor(x: np.array, type=torch.float, device=DEVICE) -> torch.Tensor:
    return torch.tensor(x, dtype=type, device=device)


def zeros(x: tuple, type=torch.float, device=DEVICE) -> torch.Tensor:
    return torch.zeros(x, dtype=type, device=device)


def run(hparams):
    start_time = time.time()
    prev_progress=0
    progress_100_cntr=0

    # load hp
    with open(HYPER_PARAMS_PATH, 'r') as file:
        default_hparams = yaml.safe_load(file)

    final_hparams = default_hparams.copy()
    final_hparams.update(hparams)
    args = munchify(final_hparams)

    run_name = (
        f"{args.environment}__{args.experiment_name}__{args.seed}__{int(time.time())}"
    )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        'hyperparameters',
        "|param|value|\n|-|-|\n%s" % (
            "\n".join(
                [f"|{key}|{value}|" for key, value in vars(args).items()]
            )
        ),
    )

    set_seed(args.seed)

    env = make_environment(args.environment)
    # agent = RandomAgent(environment=env)
    agent = PPOAgent(
        obs_space=env.observation_space,
        action_space=env.action_space
    )

    # load pretrained model (if it exists)
    pretrained_path =args.get("pretrained_model_path", None)

    if pretrained_path and os.path.exists(pretrained_path):
        agent.model.load_state_dict(torch.load(pretrained_path))
        logger.info(f"loaded pretrained model from: {pretrained_path}")
    else:
        logger.info("No pretrained model loaded")

    episode_rewards = []
    global_step = 0
    for episode in range(args.num_episodes):
        obs, _ = env.reset(seed=args.seed)
        done = False
        total_reward = 0
        episode_length = 0

        while not done:
            
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
            action, logp, val = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # print("reward-",reward)
            done = terminated or truncated

            
            agent.store(obs, action, reward, val, logp)
            total_reward += reward
            episode_length += 1

            global_step += 1
            obs = next_obs

        # update
        agent.finish_trajectory(last_val=0.0)
        agent.update()

        lap = lap_time(info)
        writer.add_scalar("Metrics/Lap_Time", lap, global_step)
        writer.add_scalar("Metrics/Total_Reward", total_reward, global_step)

        # Progress
        # progress = info.get('progress', 0)  #  reward_params
        progress = info['reward_params']['progress']
        writer.add_scalar("Metrics/Progress", progress, global_step)

        # save the full model object if progress crosses threshold
        if progress >= 75.0:
            model_name = f"{agent.name}_progress_{int(progress)}_bots.torch"
            torch.save(agent, model_name)
            logger.info(f"Saved full model as: {model_name}")
        
        if progress==100:
            if prev_progress==progress:
                progress_100_cntr+=1
            else:
                progress_100_cntr=0

        
        if progress_100_cntr>=10:
            model_name = f"{agent.name}_progress_{int(progress)}_{int(progress_100_cntr)}_bots.torch"
            torch.save(agent, model_name)
            logger.info(f"Saved full model as: {model_name}")
        
        prev_progress=progress
        

        # centerline Deviation (normalized 0-1)
        # deviation = np.mean(np.abs(info.get('distance_from_center', 0)))

        reward_params = info.get('reward_params', {})
        track_width = reward_params.get('track_width', 1.0) 
        distance_from_center = reward_params.get('distance_from_center', 0.0)

        # norm deviation
        deviation = min(
            np.abs(distance_from_center) / track_width,
            1.0 
        )

        writer.add_scalar("Metrics/Center_Deviation", deviation, global_step) 

        
        episode_rewards.append(total_reward)

        writer.add_scalar("charts/episodic_return", total_reward, episode)
        writer.add_scalar("charts/episodic_length", episode_length, episode)

        
        logger.info(
            f'Episode {episode + 1}/{args.num_episodes} - '
            f'Global Steps {global_step} - '
            f'Return: {total_reward:.2f} - '
            f'Progress: {progress}% - '
            f'Deviation: {deviation:.2f} - '
            f'Length: {episode_length} - '
            f'Lap Time: {lap:.2f} sec'
        )

        # wrire to .txt
        with open(log_txt_path, mode='a') as txt_file:
            txt_file.write(
                f'Episode {episode + 1}/{args.num_episodes} - '
                f'Global Steps {global_step} - '
                f'Return: {total_reward:.2f} - '
                f'Progress: {progress:.2f}% - '
                f'Deviation: {deviation:.2f} - '
                f'Length: {episode_length} - '
                f'Lap Time: {lap:.2f} sec\n'
            )

        # save for plotting
        episode_logs.append({
            "episode": episode + 1,
            "global_step": global_step,
            "reward": total_reward,
            "progress": progress,
            "deviation": deviation,
            "length": episode_length,
            "lap_time": lap
        })

        # sav model
    torch.save(agent, f'{agent.name}_bots.torch')
    logger.info(f'Model {agent.name}_bots saved.')

    env.close()
    writer.close()