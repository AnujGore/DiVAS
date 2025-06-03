from environment import SchedulerEnv
from policy import PPO, reward
import torch
import numpy as np
from utils import time_to_15min_index

import matplotlib.pyplot as plt
from rich.progress import track

days = 3; tasks = 4
start_hour = time_to_15min_index("07:00")
end_hour = time_to_15min_index("18:00")

mySchedule = SchedulerEnv(days, tasks)
mySchedule.fetch_info()

# Hyperparameters
epsilon_clip = 0.2
lr = 0.001
NUM_EPISODES = 1000
NUM_EPOCHS = 10
GAMMA = 0.99

policy = PPO(mySchedule, learning_rate = lr)

training_final_rewards = []
training_cumulative_rewards = []

for episode in track(range(NUM_EPISODES), description = "Episodes"):
    mySchedule = SchedulerEnv(days, tasks)
    mySchedule.fetch_info()

    #Just for one day right now
    state = mySchedule.schedule[:, 0]
    done = False
    episode_states = []
    episode_actions = []
    episode_log_probs = []
    episode_values = []
    episode_rewards = []
    idx = 0

    while not done:
        if state[idx] == 0.0:
            action_probs = torch.nn.functional.softmax(policy.actor(state), dim = 0)
            value = policy.critic(state)

            action = torch.multinomial(action_probs, 1)+2
            log_prob = torch.log(action_probs[action - 2])

            next_state, step_reward, _ = mySchedule.step(action, idx, 0)

            state = next_state
            episode_states.append(state)
            episode_actions.append(action)
            episode_log_probs.append(log_prob)
            episode_values.append(value)
            episode_rewards.append(step_reward)

        idx += 1
        if idx >= state.shape[0]:
            break

    final_reward, _ = reward(state, mySchedule.num_tasks, importance = torch.ones(mySchedule.num_tasks).numpy())
    training_cumulative_rewards.append(sum(episode_rewards))

    episode_rewards[-1] += final_reward
    training_final_rewards.append(final_reward)

    episode_states = torch.tensor(np.array(episode_states))
    episode_actions = torch.tensor(np.array(episode_actions))
    episode_log_probs = torch.tensor(episode_log_probs)
    episode_values = torch.tensor(episode_values, requires_grad=True).to(dtype = torch.float32)

    # returns = []
    # G = 0
    # for r in reversed(episode_rewards):
    #     G = r + GAMMA * G
    #     returns.insert(0, G)

    # returns = torch.tensor(returns, requires_grad=True).to(dtype = torch.float32)
    
    returns = torch.tensor([final_reward]*len(episode_values), requires_grad=True).to(dtype = torch.float32)
    advantages = returns - episode_values.squeeze()

    for _ in range(NUM_EPOCHS):
        action_probs = torch.nn.functional.softmax(policy.actor(episode_states), dim=1)
        new_log_probs = torch.log(action_probs.gather(1, episode_actions.view(-1, 1) - 2))

        ratio = torch.exp(new_log_probs - episode_log_probs)

        # Compute the clipped loss
        surrogate_loss = torch.min(ratio * advantages, torch.clamp(ratio, 1 - epsilon_clip, 1 + epsilon_clip) * advantages)
        policy_loss = -surrogate_loss.mean()

        # Update actor
        policy.actor_optim.zero_grad()
        policy_loss.backward()
        policy.actor_optim.step()

        # Update critic
        value_loss = torch.nn.functional.mse_loss(episode_values.squeeze(), returns).to(dtype = torch.float32)
        policy.critic_optim.zero_grad()
        value_loss.backward()
        policy.critic_optim.step()


#Lets see what it says
mySchedule_test = SchedulerEnv(days, tasks)
test_state = mySchedule_test.schedule[:, 0]
idx = 0; done = False
while not done:
    test_action_probs = torch.nn.functional.softmax(policy.actor(test_state), dim = 0)
    test_action = torch.multinomial(test_action_probs, 1)+2
    test_next_state = mySchedule.new_event(test_action, idx, 0)
    test_state = test_next_state
    idx += 1
    if idx >= test_state.shape[0]:
        break

print(test_state)

plt.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]
    })

x = np.arange(len(training_final_rewards))  

# Plotting the two lines
plt.plot(x, training_final_rewards, label='Final Rewards', color='dodgerblue')
plt.plot(x, training_cumulative_rewards, label='Cumulative Rewards', color='royalblue')

# Adding labels and title
plt.xlabel('Episode')
plt.ylabel('Reward')

# Adding a legend
plt.legend()
plt.grid(True)

# Display the plot
plt.show()