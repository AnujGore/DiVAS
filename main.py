from environment import SchedulerEnv
from policy import PPO, is_stale, spike_actor_network, FeedForwardNN
import torch
import numpy as np

import matplotlib.pyplot as plt
from rich.progress import track       

n_tasks = 5
n_days = 3

# ==== MAIN TRAINING ====
NUM_EPISODES = 10000
NUM_EPOCHS = 5
GAMMA = 0.99
epsilon_clip = 0.2
lr = 1e-4
best_reward = float('-inf') 
best_model_path = 'best_ppo_model.pth'

env = SchedulerEnv(days=n_days, num_tasks=n_tasks)
policy = PPO(env, learning_rate=lr)
reward_sparser = FeedForwardNN(1, [32, 64], env.schedule.shape[0])

cumulative_rewards = []

for episode in track(range(NUM_EPISODES)):
    env = SchedulerEnv(days=n_days, num_tasks=n_tasks)
    # env.fetch_info()

    states, actions, log_probs, values, rewards = [], [], [], [], []

    state_raw = env.schedule[:, 0]
    idx = 0; done = False

    while not done:
        # ---- Include slot index one-hot ----
        slot_one_hot = np.zeros(env.schedule.shape[0])
        slot_one_hot[idx] = 1
        input_state = np.concatenate([state_raw, slot_one_hot])
        input_state_torch = torch.tensor(input_state, dtype=torch.float32)

        # ---- Actor & Critic ----
        action_logits = policy.actor(input_state_torch)
        action_probs = torch.nn.functional.softmax(action_logits, dim=0)
        value = policy.critic(input_state_torch)

        # ---- Sample action ----
        action = torch.multinomial(action_probs, 1).item()  # tasks are 1-indexed
        log_prob = torch.log(action_probs[action] + 1e-8)

        # ---- Step ----
        next_state, step_reward, done = env.step(action, idx, 0, importance = torch.ones(env.num_tasks).numpy())

        states.append(input_state)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(step_reward)

        state_raw = next_state

        idx += 1

    # rewards = reward_sparser(torch.unsqueeze(reward, dim = -1))

    # ---- Compute discounted returns ----
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    values = torch.stack(values).squeeze()
    advantages = returns - values.detach()

    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions).long()
    log_probs = torch.stack(log_probs)

    # ---- PPO update ----
    for _ in range(NUM_EPOCHS):
        logits = policy.actor(states)
        new_probs = torch.nn.functional.softmax(logits, dim=1)
        new_log_probs = torch.log(new_probs.gather(1, (actions.view(-1, 1))))

        ratio = torch.exp(new_log_probs.squeeze() - log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon_clip, 1 + epsilon_clip) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        policy.actor_optim.zero_grad()
        actor_loss.backward()
        policy.actor_optim.step()

        new_values = policy.critic(states).squeeze()
        critic_loss = torch.nn.functional.mse_loss(new_values, returns)

        policy.critic_optim.zero_grad()
        critic_loss.backward()
        policy.critic_optim.step()

        # sampled_actions = torch.multinomial(new_probs, 1).squeeze(1)



    cumulative_rewards.append(sum(rewards))
    print(f"Episode {episode+1} | Reward: {sum(rewards):.2f}")

    if is_stale(cumulative_rewards):
        print("Spiking actor network due to stale rewards.")
        spike_actor_network(policy.actor)
    
    if sum(rewards) > best_reward:
        best_reward = sum(rewards)
        torch.save({
            'actor_state_dict': policy.actor.state_dict(),
            'critic_state_dict': policy.critic.state_dict(),
            'actor_optimizer_state_dict': policy.actor_optim.state_dict(),
            'critic_optimizer_state_dict': policy.critic_optim.state_dict(),
            'best_reward': best_reward,
        }, best_model_path)
        

plt.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]
    })

x = np.arange(len(cumulative_rewards))  

# Plotting the two lines
plt.plot(x, cumulative_rewards, label='Final Rewards', color='dodgerblue')

# Adding labels and title
plt.xlabel('Episode')
plt.ylabel('Reward')

# Adding a legend
plt.legend()
plt.grid(True)

# Display the plot
plt.show()

# === Test the policy ===
test_env = SchedulerEnv(days=n_days, num_tasks=n_tasks)
# test_env.fetch_info()

policy = PPO(test_env, learning_rate=lr)

checkpoint = torch.load(best_model_path)

# Load state dicts into models and optimizers
policy.actor.load_state_dict(checkpoint['actor_state_dict'])
policy.critic.load_state_dict(checkpoint['critic_state_dict'])
policy.actor_optim.load_state_dict(checkpoint['actor_optimizer_state_dict'])
policy.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])

state_raw = test_env.schedule[:, 0]
idx = 0

while idx < test_env.schedule.shape[0]:
    if state_raw[idx] == 0:
        slot_one_hot = np.zeros(test_env.schedule.shape[0])
        slot_one_hot[idx] = 1
        input_state = np.concatenate([state_raw, slot_one_hot])
        input_state_torch = torch.tensor(input_state, dtype=torch.float32)

        logits = policy.actor(input_state_torch)
        action_probs = torch.nn.functional.softmax(logits, dim=0)
        action = torch.argmax(action_probs).item() # greedy action for testing

        next_state, reward_step, _ = test_env.step(action, idx, 0, torch.ones(test_env.num_tasks).numpy())

        state_raw = next_state
    idx += 1

print(test_env.compute_continuous_counts())

print("Example Schedule")
print(test_env.schedule)
