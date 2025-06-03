import numpy as np
from google_calendar_interface import fetch_data

class SchedulerEnv:
    def __init__(self, days, num_tasks):
        self.days = days
        self.num_tasks = num_tasks
        self.schedule = np.zeros((24 * 4, days))
        self.task_assignment_counts = np.zeros(num_tasks)
        self.last_task = None
        self.consecutive = 0

    def fetch_info(self):
        self.schedule = fetch_data(self.schedule)

    def reset(self):
        self.schedule = np.zeros((24 * 4, self.days))
        self.task_assignment_counts = np.zeros(self.num_tasks)
        self.last_task = None
        self.consecutive = 0

    def step(self, action, idx, day):
        reward = 0
        done = False

        if self.schedule[idx, day] != 0:
            # Already filled, skip
            reward = -1
        else:
            self.schedule[idx, day] = action
            self.task_assignment_counts[action - 2] += 1

            reward += 1  # base reward for valid assignment

            if self.last_task == action:
                self.consecutive += 1
                reward += 0.5 * self.consecutive  # bonus for contiguity
            else:
                reward -= 0.25  # penalty for switching tasks
                self.consecutive = 0

            self.last_task = action

        if idx >= self.schedule.shape[0] - 1:
            done = True

        return self.schedule[:, day], reward, done

    def new_event(self, action, idx, day):
        self.schedule[idx, day] = action
        return self.schedule[:, day] 