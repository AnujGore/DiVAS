from environment import Scheduler
from policy import PPO, reward
import torch
from utils import time_to_15min_index

days = 3; tasks = 4
start_hour = time_to_15min_index("07:00")
end_hour = time_to_15min_index("18:00")

mySchedule = Scheduler(days, tasks)
mySchedule.fetch_info()

# Hyperparameters
gamma = 0.3
epsilon_clip = 0.2
lr = 0.001
NUM_EPISODES = 100
NUM_EPOCHS = 100

policy = PPO(mySchedule, learning_rate = lr)

for episode in range(NUM_EPISODES):
    mySchedule = Scheduler(days, tasks)
    mySchedule.fetch_info()

    #Just for one day right now
    state = mySchedule.schedule[:, 0]
    done = False
    episode_states = torch.empty(size=(mySchedule.schedule.shape[0], mySchedule.schedule.shape[0]))
    episode_data = torch.empty(size= (mySchedule.schedule.shape[0], 4))
    idx = 0; s = 0

    while not done:
        if state[idx] == 0.0:
            action_probs = torch.nn.functional.softmax(policy.actor(state), dim = 0)
            value = policy.critic(state)

            action = torch.multinomial(action_probs, 1)+2

            next_state = mySchedule.new_event(action, idx, 0)

            if idx == mySchedule.schedule.shape[0]-1:
                done = True
            else:
                idx += 1

            state = next_state
            episode_states[s] = torch.tensor(state)
            episode_data[s] = torch.tensor(action, value, action_probs[action-2])

        else:
            idx += 1

    my_reward, _ = reward(state, mySchedule.num_tasks, importance = torch.ones(mySchedule.num_tasks).numpy())

    # for _ in range(NUM_EPOCHS):



