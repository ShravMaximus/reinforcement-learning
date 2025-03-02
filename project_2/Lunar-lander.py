import gymnasium as gym
import gymnasium.envs.box2d
import csv
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
import random
import numpy as np
import time as time

#hp
states=8
actions=2
gamma=0.99 #0.7,0.8,0.90,0.95,0.99 -default
replay_buffer_batch_size=64
replay_buffer_size=int(1e6)
tau=0.005 #0.005 default [0.001,0.0025,0.005,0.008,0.01]
lr=0.0001 #0.0001 default [0.00001,0.00005,0.0001,0.0003,0.001]
num_episodes=2000
num_timesteps=1000
#tune hyper parameters change this to 1 for the hp thats needs tuning and set rest to 0
tune_gamma=0
tune_lr=0
tune_tau=0

# init
#env = gym.make("LunarLanderContinuous-v3")
env = gym.make("LunarLander-v3", continuous=True)
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

#Actor
class actor_nn(nn.Module):
    def __init__(self, states, actions):
        super().__init__()
        self.nnet=nn.Sequential(
            nn.Linear(states, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, actions),
            nn.Tanh()
        )
    def forward(self, state):
        return self.nnet(state)

#Critid
class critic_nn(nn.Module):
    def __init__(self, states, actions):
        super().__init__()
        self.nnet=nn.Sequential(
            nn.Linear(states+actions, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1) #q value
        )

    def forward(self, state, action):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)

        state_action = torch.cat([state, action], dim=-1)
        return self.nnet(state_action)


class replayBuffer:
    def __init__(self, capacity=replay_buffer_size):
        self.replay_buffer=deque(maxlen=capacity)
    # def store(self, state, action, reward, next_state, truncated,terminated):
    #     self.replay_buffer.append((state, action, reward, next_state, truncated,terminated))

    def get_sample(self, replay_buffer_batch_size):
        samp=random.sample(self.replay_buffer,replay_buffer_batch_size)
        state, action, reward, next_state, truncated,terminated=zip(*samp)
        #return (state, action, reward, next_state, truncated,terminated)

        state_tensor=torch.tensor(state, dtype=torch.float32)
        action_tensor=torch.tensor(action, dtype=torch.float32)
        reward_tensor=torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        next_state_tensor=torch.tensor(next_state, dtype=torch.float32)
        truncated_tensor=torch.tensor(truncated, dtype=torch.float32).unsqueeze(1)
        terminated_tensor = torch.tensor(terminated, dtype=torch.float32).unsqueeze(1)

        return (state_tensor,action_tensor,reward_tensor,next_state_tensor,truncated_tensor,terminated_tensor)

def noise_process(actions=2,mu=0,sigma=0.2,theta=0.15,dt=1):
    #dxt = θ(μ - xt)dt + σ sqrt(dt) dWt
    xt=np.zeros(actions)
    while True:
        dxt=theta*(mu-xt)*dt + sigma*np.sqrt(dt)*np.random.randn(actions)
        xt+=dxt
        yield xt

# Randomly initialize critic network Q(s,a|θQ) and actor µ(s|θµ) with weights θQ and θµ
# Initialize target network Q′ and µ′with weights θQ′←θQ′, θµ←θµ
# Initialize replay buffer R

critic=critic_nn(states,actions).to(torch.device('cpu'))
critic_opt = optim.Adam(critic.parameters(), lr=lr)

actor=actor_nn(states,actions).to(torch.device('cpu'))
actor_opt = optim.Adam(actor.parameters(), lr=lr)


actor_dash=actor_nn(states,actions).to(torch.device('cpu'))
critic_dash=critic_nn(states,actions).to(torch.device('cpu'))

actor_dash.load_state_dict(actor.state_dict())
critic_dash.load_state_dict(critic.state_dict())

replay_buffer=replayBuffer()

def train():
    state, action, reward, next_state, truncated, terminated = replay_buffer.get_sample(replay_buffer_batch_size)
    action_next=actor_dash(next_state)
    q_dash=critic_dash(next_state,action_next)
    episode_done=(terminated + truncated).clamp(min=0, max=1)

    target_q_val=reward+gamma * q_dash * (1-episode_done) #bellman

    pred_q_val=critic(state,action)
    critic_loss=nn.MSELoss()(pred_q_val, target_q_val)
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    actor_loss=-critic(state,actor(state)).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    with torch.no_grad():
        for dash_p, main_p in zip(critic_dash.parameters(), critic.parameters()):
            dash_p.data.copy_(tau * main_p.data + (1 - tau) * dash_p.data)

        for dash_p, main_p in zip(actor_dash.parameters(), actor.parameters()):
            dash_p.data.copy_(tau * main_p.data + (1 - tau) * dash_p.data)


##EXECUTE EVERYTHING BEFORE THIS

###Train the model

start_time = time.time()
training_rewards=[]
for episode in range(num_episodes):
    state,_ = env.reset()
    ou_noise=noise_process() #reset the noise
    tot_reward = 0
    for timestep in range(num_timesteps):
        tensor_state=torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action=actor(tensor_state).detach().numpy()[0]
        #print("action",action)
        action+=next(ou_noise)
        #print("action2", action)
        action = np.clip(action, -1, +1).astype(np.float64)
        #print("action3", action)
        next_state, reward, terminated, truncated, _ = env.step(action) #eturn np.array(state, dtype=np.float32), reward, terminated, False, {}
        replay_buffer.replay_buffer.append((state, action, reward, next_state, truncated,terminated))
        if len(replay_buffer.replay_buffer) >= replay_buffer_batch_size:
            train()

        state=next_state
        tot_reward+=reward

        if terminated or truncated:
            break
    training_rewards.append(tot_reward)
    print(f"Episode {episode + 1}: Reward = {tot_reward}")

env.close()
print("Training complete!")
end_time = time.time()
print("time take=",end_time-start_time)
# Save the trained models
torch.save(actor.state_dict(), "ddpg_actor.pth")
torch.save(critic.state_dict(), "ddpg_critic.pth")

#plot the training rewards

plt.figure(figsize=(10, 5))
plt.plot(training_rewards, label="Training Rewards", color='blue')
plt.axhline(y=200, color='r', linestyle='--', label="Success Threshold (200)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Rewards Per Episode")
plt.legend()
plt.grid(True,linestyle="--", alpha=0.6)
plt.savefig("training_rewards_plot.png")
#plt.show()

env.close()
print("Training complete and model saved")


##### Test
#
import gymnasium as gym
import torch
import numpy as np

# Load the trained actor model
actor = actor_nn(states, actions).to(torch.device("cpu"))
actor.load_state_dict(torch.load("ddpg_actor.pth"))
actor.eval()  # Set model to evaluation mode

csv_filename = "ddpg_test_results.csv"

testing_rewards=[]

# Init
#env = gym.make("LunarLanderContinuous-v3")
env = gym.make("LunarLander-v3", continuous=True)
num_test_episodes = 100  # # of episodes to test
results = []
max_timesteps = 1000  # steps per episode
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Episode", "Reward"])

    for episode in range(num_test_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(max_timesteps):
            with torch.no_grad():
                action = actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).numpy()[0]


            next_state, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            state = next_state

            if terminated or truncated:
                break

        testing_rewards.append(episode_reward)
        print(f"test Episode {episode + 1}: Reward = {episode_reward}")

        writer.writerow([episode + 1, episode_reward])
        results.append((episode + 1, episode_reward))

plt.figure(figsize=(10, 5))
plt.plot(testing_rewards, label="Test Rewards", color='green')
plt.axhline(y=200, color='r', linestyle='--', label="Success Threshold (200)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Test Rewards Per Episode")
plt.legend(loc="lower left")
plt.grid(True,linestyle="--", alpha=0.6)
plt.savefig("test_rewards_plot.png")

env.close()
print("testing complete!")

##HYPER PARAMETER TUNING FUNCTION
def hp_tuning(gamma,tau,lr,tune_gamma,tune_tau,tune_lr):
    import gymnasium as gym
    import gymnasium.envs.box2d
    import csv
    import matplotlib.pyplot as plt

    import torch
    import torch.nn as nn
    import torch.optim as optim

    from collections import deque
    import random
    import numpy as np
    import time as time

    # init
    #env = gym.make("LunarLanderContinuous-v3")
    env = gym.make("LunarLander-v3", continuous=True)
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    #Actor
    class actor_nn(nn.Module):
        def __init__(self, states, actions):
            super().__init__()
            self.nnet=nn.Sequential(
                nn.Linear(states, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, actions),
                nn.Tanh()
            )
        def forward(self, state):
            return self.nnet(state)

    #Critid
    class critic_nn(nn.Module):
        def __init__(self, states, actions):
            super().__init__()
            self.nnet=nn.Sequential(
                nn.Linear(states+actions, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1) #q value
            )

        def forward(self, state, action):
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32)

            state_action = torch.cat([state, action], dim=-1)
            return self.nnet(state_action)


    class replayBuffer:
        def __init__(self, capacity=replay_buffer_size):
            self.replay_buffer=deque(maxlen=capacity)
        # def store(self, state, action, reward, next_state, truncated,terminated):
        #     self.replay_buffer.append((state, action, reward, next_state, truncated,terminated))

        def get_sample(self, replay_buffer_batch_size):
            samp=random.sample(self.replay_buffer,replay_buffer_batch_size)
            state, action, reward, next_state, truncated,terminated=zip(*samp)
            #return (state, action, reward, next_state, truncated,terminated)

            state_tensor=torch.tensor(state, dtype=torch.float32)
            action_tensor=torch.tensor(action, dtype=torch.float32)
            reward_tensor=torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
            next_state_tensor=torch.tensor(next_state, dtype=torch.float32)
            truncated_tensor=torch.tensor(truncated, dtype=torch.float32).unsqueeze(1)
            terminated_tensor = torch.tensor(terminated, dtype=torch.float32).unsqueeze(1)

            return (state_tensor,action_tensor,reward_tensor,next_state_tensor,truncated_tensor,terminated_tensor)

    def noise_process(actions=2,mu=0,sigma=0.2,theta=0.15,dt=1):
        #dxt = θ(μ - xt)dt + σ sqrt(dt) dWt
        xt=np.zeros(actions)
        while True:
            dxt=theta*(mu-xt)*dt + sigma*np.sqrt(dt)*np.random.randn(actions)
            xt+=dxt
            yield xt

    # Randomly initialize critic network Q(s,a|θQ) and actor µ(s|θµ) with weights θQ and θµ
    # Initialize target network Q′ and µ′with weights θQ′←θQ′, θµ←θµ
    # Initialize replay buffer R

    critic=critic_nn(states,actions).to(torch.device('cpu'))
    critic_opt = optim.Adam(critic.parameters(), lr=lr)

    actor=actor_nn(states,actions).to(torch.device('cpu'))
    actor_opt = optim.Adam(actor.parameters(), lr=lr)


    actor_dash=actor_nn(states,actions).to(torch.device('cpu'))
    critic_dash=critic_nn(states,actions).to(torch.device('cpu'))

    actor_dash.load_state_dict(actor.state_dict())
    critic_dash.load_state_dict(critic.state_dict())

    replay_buffer=replayBuffer()

    def train():
        state, action, reward, next_state, truncated, terminated = replay_buffer.get_sample(replay_buffer_batch_size)
        action_next=actor_dash(next_state)
        q_dash=critic_dash(next_state,action_next)
        episode_done=(terminated + truncated).clamp(min=0, max=1)

        target_q_val=reward+gamma * q_dash * (1-episode_done) #bellman

        pred_q_val=critic(state,action)
        critic_loss=nn.MSELoss()(pred_q_val, target_q_val)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        actor_loss=-critic(state,actor(state)).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        with torch.no_grad():
            for dash_p, main_p in zip(critic_dash.parameters(), critic.parameters()):
                dash_p.data.copy_(tau * main_p.data + (1 - tau) * dash_p.data)

            for dash_p, main_p in zip(actor_dash.parameters(), actor.parameters()):
                dash_p.data.copy_(tau * main_p.data + (1 - tau) * dash_p.data)

    ###Train the model

    start_time = time.time()
    training_rewards=[]
    for episode in range(num_episodes):
        state,_ = env.reset()
        ou_noise=noise_process() #reset the noise
        tot_reward = 0
        for timestep in range(num_timesteps):
            tensor_state=torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action=actor(tensor_state).detach().numpy()[0]
            #print("action",action)
            action+=next(ou_noise)
            #print("action2", action)
            action = np.clip(action, -1, +1).astype(np.float64)
            #print("action3", action)
            next_state, reward, terminated, truncated, _ = env.step(action) #eturn np.array(state, dtype=np.float32), reward, terminated, False, {}
            replay_buffer.replay_buffer.append((state, action, reward, next_state, truncated,terminated))
            if len(replay_buffer.replay_buffer) >= replay_buffer_batch_size:
                train()

            state=next_state
            tot_reward+=reward

            if terminated or truncated:
                break
        training_rewards.append(tot_reward)
        print(f"Episode {episode + 1}: Reward = {tot_reward}")

    env.close()
    print("Training in hp tuning complete!")
    end_time = time.time()
    print("time take=",end_time-start_time)
    # Save the trained models
    torch.save(actor.state_dict(), "ddpg_actor.pth")
    torch.save(critic.state_dict(), "ddpg_critic.pth")

    env.close()
    print("Training complete")

    ##### Test
    #
    import gymnasium as gym
    import torch
    import numpy as np

    # Load the trained actor model
    actor = actor_nn(states, actions).to(torch.device("cpu"))
    actor.load_state_dict(torch.load("ddpg_actor.pth"))
    actor.eval()  # Set model to evaluation mode

    if tune_gamma==1:
        csv_filename = "ddpg_test_results_gamma_"+str(gamma)+".csv"
    elif tune_lr==1:
        csv_filename = "ddpg_test_results_lr_"+str(lr)+".csv"
    elif tune_tau==1:
        csv_filename = "ddpg_test_results_tau_"+str(tau)+".csv"

    testing_rewards=[]

    # Initialize the environment for testing
    #env = gym.make("LunarLanderContinuous-v3")
    env = gym.make("LunarLander-v3", continuous=True)

    num_test_episodes = 100  # Number of episodes to test
    results = []
    max_timesteps = 1000  # Max steps per episode
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Reward"])  # Write header

        for episode in range(num_test_episodes):
            state, _ = env.reset()
            episode_reward = 0

            for t in range(max_timesteps):
                with torch.no_grad():
                    action = actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).numpy()[0]

                # Execute the action
                next_state, reward, terminated, truncated, _ = env.step(action)

                episode_reward += reward
                state = next_state

                if terminated or truncated:
                    break

            testing_rewards.append(episode_reward)
            print(f"Test Episode {episode + 1}: Reward = {episode_reward}")

            writer.writerow([episode + 1, episode_reward])  # Save results in CSV
            results.append((episode + 1, episode_reward))

    env.close()
    print("Testing in hp tuning complete!")
#
# gamma=0.99 #0.7,0.8,0.90,0.95,0.99 -default
# replay_buffer_batch_size=64
# replay_buffer_size=int(1e6)
# tau=0.005 #0.005 default [0.001,0.0025,0.005,0.008,0.01]
# lr=0.0001 #0.0001 default [0.00001,0.00005,0.0001,0.0003,0.001]

##HYPER PARAMETER TUNING CHOOSE any HP or execute all

for gamma in [0.7,0.8,0.90,0.95,0.99]:
    tune_gamma=1
    tune_tau=0
    tune_lr=0
    tau = 0.005
    lr = 0.0001
    hp_tuning(gamma,tau,lr,tune_gamma,tune_tau,tune_lr)

for tau in [0.001,0.0025,0.005,0.008,0.01]:
    tune_gamma = 0
    tune_tau = 1
    tune_lr = 0
    gamma=0.99
    lr = 0.0001
    hp_tuning(gamma, tau, lr, tune_gamma, tune_tau, tune_lr)

for lr in [0.00001,0.00005,0.0001,0.0003,0.001,0.0025,0.005,0.008,0.01]:
    tune_gamma = 0
    tune_tau = 0
    tune_lr = 1
    gamma=0.99
    tau = 0.005

    hp_tuning(gamma, tau, lr, tune_gamma, tune_tau, tune_lr)



