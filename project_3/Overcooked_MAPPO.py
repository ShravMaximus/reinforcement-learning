import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from collections import deque
import random
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

for LAYOUT in ["cramped_room", "coordination_ring","counter_circuit_o_1order"]:
    print(LAYOUT)
    # hyperparameters
    SEED = 42
    #LAYOUT = "coordination_ring"
    HORIZON = 400
    if LAYOUT=="counter_circuit_o_1order":
        NUM_EPISODES = 50000
    else:
        NUM_EPISODES = 35000

    GAMMA = 0.90
    CLIP_EPSILON = 0.01 #0.2 cramped room
    LR = 3e-4 #3e-4

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


    class ActorCritic(nn.Module):
        def __init__(self, input_size,action_size):
            super(ActorCritic, self).__init__()
            self.base = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.GELU(),
            )
            self.actor = nn.Linear(256,action_size)
            self.critic = nn.Linear(256,1)

        def forward(self, x):
            x = self.base(x)
            return torch.softmax(self.actor(x), dim=-1),self.critic(x)


    class MAPPOAgent:
        def __init__(self, input_size, action_size):
            self.policy = ActorCritic(input_size, action_size)
            self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
            self.running_mean = 0
            self.running_var = 1
            self.count = 1e-4

        def normalize(self, x):
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0)
            batch_count = x.shape[0]

            delta = batch_mean- self.running_mean
            total_count = self.count+ batch_count
            self.running_mean = (self.count * self.running_mean +batch_count * batch_mean)/ total_count
            self.running_var = (self.count * self.running_var +batch_count * batch_var +
                                delta ** 2 * self.count * batch_count / total_count)/ total_count
            self.count = total_count

            return (x - self.running_mean) / torch.sqrt(self.running_var+ 1e-8)

        def get_action(self, state, greedy=False):
            with torch.no_grad():
                state = torch.FloatTensor(state)
                state = self.normalize(state)
                probs, value = self.policy(state)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample() if not greedy else torch.argmax(probs)
                return action.item(), dist.log_prob(action), value.item()

        def compute_gae(self, rewards, values, dones):
            advantages = []
            last_adv = 0
            next_value = 0
            gae_lambda=0.95


            rewards = np.array(rewards, dtype=np.float32)
            values = np.array(values, dtype=np.float32)
            dones = np.array(dones, dtype=np.float32)

            for t in reversed(range(len(rewards))):
                mask = 1 -dones[t]
                delta = rewards[t] + GAMMA * next_value * mask -values[t]
                last_adv = delta + GAMMA * gae_lambda * mask *last_adv
                advantages.insert(0,last_adv)
                next_value = values[t]

            return np.array(advantages, dtype=np.float32)

        def update(self, states, actions, old_log_probs, returns, advantages):
            entropy_coeff=0.01
            epochs=3
            critic_coeff=0.5
            grad_norm=0.5

            states = np.array(states, dtype=np.float32)
            states = torch.tensor(states, dtype=torch.float32)
            states = self.normalize(states)

            actions = torch.tensor(np.array(actions), dtype=torch.long)
            old_log_probs = torch.tensor(np.array(old_log_probs), dtype=torch.float32)
            returns = torch.tensor(np.array(returns), dtype=torch.float32)
            advantages = torch.tensor(np.array(advantages), dtype=torch.float32)

            # normlaize adv
            advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)

            dataset = MAPPO_Dataset(states, actions, old_log_probs, returns, advantages)
            dataloader_mini_batch = DataLoader(dataset, batch_size=256, shuffle=True)

            for _ in range(epochs):
                for batch in dataloader_mini_batch:
                    s, a, old_lp, ret, gae_adv = batch
                    probs, values = self.policy(s)
                    dist = torch.distributions.Categorical(probs)
                    new_log_probs = dist.log_prob(a)
                    entropy = dist.entropy().mean()

                    # policy_loss
                    prob_ratio = (new_log_probs - old_lp).exp()
                    clip_adv = torch.clamp(prob_ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * gae_adv
                    policy_loss = -torch.min(prob_ratio * gae_adv, clip_adv).mean()

                    # valu_loss
                    value_loss = 0.5 * (ret - values.squeeze()).pow(2).mean()

                    # combine_loss
                    loss = policy_loss + critic_coeff * value_loss - entropy_coeff * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), grad_norm)
                    self.optimizer.step()


    class MAPPO_Dataset(Dataset):
        def __init__(self, states, actions, old_log_probs, returns, advantages):
            self.states = states
            self.actions = actions
            self.old_log_probs = old_log_probs
            self.returns = returns
            self.advantages = advantages

        def __len__(self):
            return len(self.states)

        def __getitem__(self, idx):
            return (self.states[idx],
                    self.actions[idx],
                    self.old_log_probs[idx],
                    self.returns[idx],
                    self.advantages[idx])



    reward_shaping = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0.1
    }

    mdp = OvercookedGridworld.from_layout_name(LAYOUT, rew_shaping_params=reward_shaping)
    base_env = OvercookedEnv.from_mdp(mdp, horizon=HORIZON, info_level=0)
    env = gym.make("Overcooked-v0", base_env=base_env,featurize_fn=base_env.featurize_state_mdp, disable_env_checker=True)

    agents = [MAPPOAgent(96, 6), MAPPOAgent(96, 6)]
    episode_soups = []
    avg_100_soups = []
    avg_100_episodes=[]
    test_100_episode_number=[]
    test_100_episode_soups=[]

    episode_onion_pickups = []
    episode_dish_pickups = []

    avg_100_onion_pickups = []
    avg_100_dish_pickups =  []

    test_onion_pickups = []
    test_dish_pickups = []

    train_onion_pickups=[]
    train_dish_pickups=[]

    for episode in range(NUM_EPISODES + 102):
        state = env.reset()
        done = False
        ep_reward = 0
        ep_soups = 0
        batch = {agent: [] for agent in range(2)}

        testing_mode = episode >= NUM_EPISODES

        while not done:
            states = state["both_agent_obs"]
            actions = []
            log_probs = []
            values = []

            for agent_idx in [0, 1]:
                a, lp, v = agents[agent_idx].get_action(states[agent_idx])
                actions.append(a)
                log_probs.append(lp)
                values.append(v)

            next_state, reward, done, info = env.step(actions)
            #shaped_rewards = info["shaped_r_by_agent"]
            shaped_rewards = [
                info["shaped_r_by_agent"][i] + (reward / 2)  # half of sparse reward to each agent
                for i in [0, 1]]
            ep_soups += int(reward / 20)

            # # track actions (basic reward-shaping keys)
            # for agent_idx in [0, 1]:
            #     r = shaped_rewards[agent_idx]
            #     if r == 3:
            #         action_counts["put_in_pot"] += 1
            #     elif r == 5:
            #         action_counts["soup_pickup"] += 1
            #     elif r == 20:
            #         action_counts["soup_delivered"] += 1
            #     elif "PICKED" in info.get("events", [""])[agent_idx].upper():
            #         action_counts["onion_pickup"] += 1

            if not testing_mode:
                for agent_idx in [0, 1]:
                    transition = (
                        states[agent_idx],
                        actions[agent_idx],
                        log_probs[agent_idx],
                        shaped_rewards[agent_idx],
                        values[agent_idx],
                        done
                    )
                    batch[agent_idx].append(transition)

            state = next_state
            ep_reward += reward

        if not testing_mode:

            for agent_idx in [0, 1]:
                states_b, actions_b, log_probs_b, rewards_b, values_b, dones_b = zip(*batch[agent_idx])

                rewards = np.array(rewards_b)
                values = np.array(values_b)
                dones = np.array(dones_b)

                advantages = agents[agent_idx].compute_gae(rewards, values, dones) #gae
                returns = advantages + values

                #  update
                agents[agent_idx].update(
                    states_b,
                    actions_b,
                    log_probs_b,
                    returns,
                    advantages
                )

        episode_soups.append(ep_soups)

        game_stats = info["episode"]["ep_game_stats"]
        onion_pickup = len(game_stats["onion_pickup"][0]) + len(game_stats["onion_pickup"][1])
        dish_pickup = len(game_stats["dish_pickup"][0]) + len(game_stats["dish_pickup"][1])

        episode_onion_pickups.append(onion_pickup)
        episode_dish_pickups.append(dish_pickup)

        if testing_mode: #save episodes and rewards
            test_100_episode_number.append(episode - NUM_EPISODES + 1)
            test_100_episode_soups.append(ep_soups)

            test_onion_pickups.append(onion_pickup)
            test_dish_pickups.append(dish_pickup)

        # model saving
        if (episode+1) % 100 == 0:
            avg_soups = np.mean(episode_soups[-100:])
            avg_onion_pickup = np.mean(episode_onion_pickups[-100:])
            avg_dish_pickup = np.mean(episode_dish_pickups[-100:])

            if (episode + 1) <= NUM_EPISODES:
                print(f"Training Episode {episode+1} | Last 100 Avg Soups: {avg_soups:.2f}")
                avg_100_soups.append(avg_soups)
                avg_100_episodes.append(episode + 1)

                avg_100_onion_pickups.append(avg_onion_pickup)
                avg_100_dish_pickups.append(avg_dish_pickup)

            elif (episode + 1) == NUM_EPISODES + 100:
                print("Testing episode: ", episode + 1)
                print(f"Testing Episode | 100 Avg Soups: {avg_soups:.2f}")
                test_result=(f"Testing Episode | Avg Soups delivered over 100 consecutive episodes: {avg_soups:.2f}\n"
                            f"→ Avg Onion Pickups: {avg_onion_pickup:.2f} | Avg Dish Pickups: {avg_dish_pickup:.2f}")

                with open(f"test_results_{LAYOUT}.txt", "a") as file:
                    file.write(test_result + "\n")




    if LAYOUT == "cramped_room":
        avg_100_episodes_cramped_room,avg_100_soups_cramped_room=avg_100_episodes, avg_100_soups
        test_100_episodes_cramped_room, test_100_soups_cramped_room=test_100_episode_number,test_100_episode_soups

        avg_100_onion_pickups_cramped_room = avg_100_onion_pickups
        avg_100_dish_pickups_cramped_room = avg_100_dish_pickups


        test_100_onion_pickups_cramped_room = test_onion_pickups
        test_100_dish_pickups_cramped_room = test_dish_pickups


    elif LAYOUT == "coordination_ring":
        avg_100_episodes_coordination_ring, avg_100_soups_coordination_ring = avg_100_episodes, avg_100_soups
        test_100_episodes_coordination_ring, test_100_soups_coordination_ring=test_100_episode_number,test_100_episode_soups

        avg_100_onion_pickups_coordination_ring = avg_100_onion_pickups
        avg_100_dish_pickups_coordination_ring = avg_100_dish_pickups

        test_100_onion_pickups_coordination_ring = test_onion_pickups
        test_100_dish_pickups_coordination_ring = test_dish_pickups


    elif LAYOUT == "counter_circuit_o_1order":
        avg_100_episodes_counter_circuit, avg_100_soups_counter_circuit = avg_100_episodes, avg_100_soups
        test_100_episodes_counter_circuit, test_100_soups_counter_circuit=test_100_episode_number,test_100_episode_soups

        avg_100_onion_pickups_counter_circuit = avg_100_onion_pickups
        avg_100_dish_pickups_counter_circuit = avg_100_dish_pickups

        test_100_onion_pickups_counter_circuit = test_onion_pickups
        test_100_dish_pickups_counter_circuit = test_dish_pickups

#plot combined soups

plt.figure(figsize=(12, 5))
plt.plot(avg_100_episodes_cramped_room, avg_100_soups_cramped_room, linewidth=2, color='blue', label="Cramped Room")
plt.plot(avg_100_episodes_coordination_ring, avg_100_soups_coordination_ring, linewidth=2, color='orange', label="Coordination Ring")
plt.plot(avg_100_episodes_counter_circuit, avg_100_soups_counter_circuit, linewidth=2, color='green', label="Counter Circuit 0-1 Order")

plt.axhline(y=7, color='red', linestyle='--', linewidth=2, label="Threshold (7 Soups)")

plt.xlabel("Training Episode Number")
plt.ylabel("Average Soups per 100 Episodes")
plt.title("Training Progres Across Three Overcooked Layouts: Average Soups per 100 Episodes")


plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("combined_avg_soups_per_100_episodes.png")

#train onion pickup

plt.figure(figsize=(12, 5))

plt.plot(avg_100_episodes_cramped_room, avg_100_onion_pickups_cramped_room, linewidth=2, color='blue', label="Cramped Room")
plt.plot(avg_100_episodes_coordination_ring, avg_100_onion_pickups_coordination_ring, linewidth=2, color='orange',label="Coordination Ring")
plt.plot(avg_100_episodes_counter_circuit, avg_100_onion_pickups_counter_circuit, linewidth=2, color='green',label="Counter Circuit 0-1 Order")

plt.xlabel("Training Episode Number")
plt.ylabel("Average Onion pickups per 100 Episodes")
plt.title("Training Progres Across Three Overcooked Layouts: Average Onion pickups per 100 Episodes")

plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("combined_avg_onion_per_100_episodes.png")

# train dish pickup

plt.figure(figsize=(12, 5))

plt.plot(avg_100_episodes_cramped_room, avg_100_dish_pickups_cramped_room, linewidth=2, color='blue',label="Cramped Room")
plt.plot(avg_100_episodes_coordination_ring, avg_100_dish_pickups_coordination_ring, linewidth=2, color='orange',label="Coordination Ring")
plt.plot(avg_100_episodes_counter_circuit, avg_100_dish_pickups_counter_circuit, linewidth=2, color='green',label="Counter Circuit 0-1 Order")

plt.xlabel("Training Episode Number")
plt.ylabel("Average Dish pickups per 100 Episodes")
plt.title("Training Progres Across Three Overcooked Layouts: Average Dish pickups per 100 Episodes")

plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("combined_avg_dish_per_100_episodes.png")


##### Test

###COMBINED TEST SOUPS

plt.figure(figsize=(12, 5))
plt.plot(test_100_episode_number[:100], test_100_soups_cramped_room[:100], linewidth=2, color='blue', label="Cramped Room")
plt.plot(test_100_episode_number[:100], test_100_soups_coordination_ring[:100], linewidth=2, color='orange', label="Coordination Ring")
plt.plot(test_100_episode_number[:100], test_100_soups_counter_circuit[:100], linewidth=2, color='green', label="Counter Circuit 0-1 Order")

plt.axhline(y=7, color='red', linestyle='--', linewidth=2, label="Threshold (7 Soups)")

plt.xlabel("Test Episode Number")
plt.ylabel("Soups per Episodes")
plt.title("Agent Performance Across Three Overcooked Layouts: Soups Delivered 100 Episode Evaluation")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("combine_test_soups_100_episodes.png")

#Test Onion

plt.figure(figsize=(12, 5))
plt.plot(test_100_episode_number[:100], test_100_onion_pickups_cramped_room[:100], linewidth=2, color='blue',
         label="Cramped Room")
plt.plot(test_100_episode_number[:100], test_100_onion_pickups_coordination_ring[:100], linewidth=2, color='orange',
         label="Coordination Ring")
plt.plot(test_100_episode_number[:100],test_100_onion_pickups_counter_circuit[:100], linewidth=2, color='green',
         label="Counter Circuit 0-1 Order")

plt.xlabel("Test Episode Number")
plt.ylabel("Onion pickups per Episodes")
plt.title("Agent Performance Across Three Overcooked Layouts: Onion Pickups 100 Episode Evaluation")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("combine_test_onion_100_episodes.png")

plt.figure(figsize=(12, 5))
plt.plot(test_100_episode_number[:100], test_100_dish_pickups_cramped_room[:100], linewidth=2, color='blue',
         label="Cramped Room")
plt.plot(test_100_episode_number[:100], test_100_dish_pickups_coordination_ring[:100], linewidth=2, color='orange',
         label="Coordination Ring")
plt.plot(test_100_episode_number[:100], test_100_dish_pickups_counter_circuit[:100], linewidth=2, color='green',
         label="Counter Circuit 0-1 Order")

plt.xlabel("Test Episode Number")
plt.ylabel("Dish pickups per Episodes")
plt.title("Agent Performance Across Three Overcooked Layouts: Dish Pickups 100 Episode Evaluation")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("combine_test_dish_100_episodes.png")

