import numpy as np
from google_calendar_interface import fetch_data
import torch

class SchedulerEnv:
    def __init__(self, days, num_tasks):
        self.days = days
        self.num_tasks = num_tasks
        self.schedule = np.zeros((24 * 4, days))

    # def fetch_info(self):
    #     # self.schedule = fetch_data(self.schedule)
    #     self.schedule = np

    def reset(self):
        self.schedule = np.zeros((24 * 4, self.days))

    def step(self, action, idx, day, importance):
        reward = 0
    
        self.schedule[idx, day] = action+1

        done = np.all(self.schedule[:, day] != 0)
        # if done:
        count, continuous = self.compute_continuous_counts()

        target_distribution = torch.nn.functional.softmax(torch.tensor(importance), dim=0)
        predicted_distribution_count = torch.nn.functional.softmax(torch.tensor(count), dim=0)

        count_error = (1/sum(np.abs(target_distribution - predicted_distribution_count)))
        continuous_error = (1/sum(np.abs(target_distribution*self.schedule.shape[0] - continuous)))

        reward += count_error #+ continuous_error


        return self.schedule[:, day], reward, done

    def compute_continuous_counts(self):
        all_count = np.zeros(shape = (self.days, self.num_tasks))
        all_continuous = np.zeros_like(all_count)
        for i in range(self.days):
            this_schedule = self.schedule[:, i]
            continuous = np.zeros(shape = (self.num_tasks))
            count  = np.zeros_like(continuous)
            for task in range(self.num_tasks):
                task_idx = np.where(this_schedule == task+1)[0]

                counter = 0; max_counter = 0
                for pos, _ in enumerate(task_idx[:-1]):
                    if task_idx[pos]+1 == task_idx[pos+1]:
                        counter += 1
                    else:
                        if max_counter < counter: 
                            max_counter = counter
                        counter = 0

                continuous[task] = max_counter
                count[task] = len(task_idx)

            all_count[i] = count
            all_continuous[i] = continuous

        return all_count.max(axis = 0), all_continuous.max(axis = 0)


    def new_event(self, action, idx, day):
        self.schedule[idx, day] = action
        return self.schedule[:, day] 