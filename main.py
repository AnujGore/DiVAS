import numpy as np
import torch
from utils import summarize_day_array

num_tasks = 3
task_importance = torch.tensor([8, 5, 3]).float() #Out of 10

task_ids = [i+1 for i in range(num_tasks)]

start_time = 9
end_time = 18

hours_worked = end_time-start_time
day_array = np.zeros(shape = (4*hours_worked))

#Blocking
day_array[13:15] = -1 #Lunch


task_importance_norm = torch.nn.functional.normalize(task_importance, p=1.0, dim = 0)
task_importance_count = torch.round((4*hours_worked)*task_importance_norm)

counter = 0; head = 0
for slot in range(day_array.size):
    if day_array[slot] == 0 :
        day_array[slot] = task_ids[head]
        counter += 1

    if np.isclose(counter, task_importance_count[head]):
        counter = 0
        head+=1

summarize_day_array(day_array)
