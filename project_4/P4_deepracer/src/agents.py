import abc
import torch
import torch.nn as nn

from src.transforms import EncodeObservation

import torch.optim as optim
import numpy as np
import random
from collections import deque


class Agent(nn.Module, abc.ABC):
    '''Boilerplate class for providing interface'''

    def __init__(self, name):
        super().__init__()
        self.name = name

    @abc.abstractmethod
    def get_action(self, x):
        raise NotImplementedError


class RandomAgent(Agent):
    '''
    A random agent for demonstrating usage of the environment
    '''

    def __init__(self, environment, name='random'):
        super().__init__(name=name)
        self.action_space = environment.action_space

    def get_action(self, x):
        return self.action_space.sample()


class MyFancyAgent(Agent):
    '''
    Your own deepracer agent.
    '''

    def __init__(self, name='my_fancy_agent'):
        super().__init__(name=name)

        # in case you want to modify/ change/ transform
        # the observation (of course we have to un-flatten it).
        self.encoder = EncodeObservation()

        # anything else you may need...
        raise NotImplementedError

    def get_action(self, x):
        # un-flatten observation, and other transforms if required.
        z = self.encoder(x)

        # implement your fancy policy
        raise NotImplementedError


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),  
            nn.ReLU(),
        )
        self.policy =nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256), 
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(256, 256),
            #nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        shared = self.shared(x)

        return self.policy(shared), self.value(shared).squeeze(-1)

    def get_action(self, obs):
        logits, value =self.forward(obs)
        dist =torch.distributions.Categorical(logits=logits)
        action= dist.sample()
        log_prob= dist.log_prob(action)
        return action, log_prob, value


class PPOAgent(Agent):
    def __init__(self, obs_space, action_space, name="ppo_agent", gamma=0.90, lam=0.95, clip_ratio=0.01, lr=3e-4, epochs=3, batch_size=64,ent_coef=0.01):
        super().__init__(name=name)
        self.ent_coef = ent_coef
        self.obs_dim= int(np.prod(obs_space.shape))
        self.act_dim= action_space.n

        self.gamma =gamma
        self.lam =lam
        self.clip_ratio= clip_ratio
        self.epochs =epochs
        self.batch_size= batch_size

        self.model= ActorCritic(self.obs_dim, self.act_dim)
        self.optimizer= optim.Adam(self.model.parameters(), lr=lr)

        self.reset_trajectory()

    def reset_trajectory(self):
        self.obs_buf =[]
        self.act_buf =[]
        self.adv_buf =[]
        self.rew_buf =[]
        self.ret_buf =[]
        self.logp_buf =[]
        self.val_buf = []

    def get_action(self, obs):
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, log_prob, value =self.model.get_action(obs_tensor)
        return action.item(),log_prob.item(),value.item()

    def store(self, obs, act, rew, val, logp):
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)

        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.val_buf.append(val)
        self.logp_buf.append(logp)

    def finish_trajectory(self, last_val=0):
        rews = np.append(self.rew_buf,last_val)
        vals = np.append(self.val_buf,last_val)

        deltas = rews[:-1] + self.gamma *vals[1:] - vals[:-1]
        adv = self.discount_cumsum(deltas,self.gamma * self.lam)
        ret = self.discount_cumsum(rews,self.gamma)[:-1]

        self.adv_buf = adv
        #self.adv_buf = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
        self.ret_buf = ret

    def discount_cumsum(self, x, discount):
        return np.array([sum(discount ** i * x[i + j] for i in range(len(x) - j)) for j in range(len(x))])

    def update(self):
        obs =torch.tensor(np.array(self.obs_buf), dtype=torch.float32)
        act =torch.tensor(self.act_buf, dtype=torch.int64)
        adv =torch.tensor(self.adv_buf, dtype=torch.float32)
        ret =torch.tensor(self.ret_buf, dtype=torch.float32)
        logp_old =torch.tensor(self.logp_buf, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(self.epochs):
            idxs = np.arange(len(obs))
            np.random.shuffle(idxs)

            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_idx = idxs[start:end]

                batch_obs =obs[batch_idx]
                batch_act =act[batch_idx]
                batch_adv =adv[batch_idx]
                batch_ret =ret[batch_idx]
                batch_logp_old =logp_old[batch_idx]

                logits, value =self.model(batch_obs)
                dist =torch.distributions.Categorical(logits=logits)
                logp =dist.log_prob(batch_act)

                entropy =dist.entropy().mean()

                ratio =torch.exp(logp - batch_logp_old)

                clipped= torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                policy_loss = -torch.min(ratio * batch_adv, clipped * batch_adv).mean()
                value_loss= nn.MSELoss()(value.squeeze(), batch_ret)

                #loss = policy_loss + 0.5 *value_loss

                loss = policy_loss + 0.5 * value_loss -self.ent_coef *entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),0.5)
                self.optimizer.step()

        self.reset_trajectory()




