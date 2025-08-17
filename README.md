# DiVAS.

## Overview
In this project, I am trying to create a software that optimizes my daily schedule. The background theory is based on neurobiology - No screen 1 hour before sleep, etc. The flow is that I'll probably download TinyLlama (coz of space) and use DPO to fine tune it for neurobiology. A sanity check would be to ask it so solve a basic physics questions and it SHOULD fail.

There are two key components - 
    1. Google Calendar Binding (done)
    2. vanilla, uninformed RL on current blank schedule (doing)
    3. DPO on the LLM (tbd)


The higher-level architecture would be like this

         +------------+       +------------+       +-------------+
         |  Day In    | ----> |  RL Model  | ----> |   Day Out   |
         | (Schedule) |       |   (PPO)    |       | (Optimized) |
         +------------+       +------------+       +-------------+
                                   ^
                                   |
                             +-------------+
                             |     LLM     |
                             |  (Advisor)  |
                             +-------------+

## Stage 1:

Using an ILP solution. Only depends on priorities now. Some NLP would be required to understand the project and how long and when to work on the project. 

### Usage

```python
from src.environment import SchedulerEnv
from src.ilp_solution import schedule_projects_daily
from calendar_interface.google_calendar_interface import write_to_calendar

n_tasks = 5
n_days = 1

env = SchedulerEnv(days=n_days, num_tasks=n_tasks, hours = ("06:00", "18:00"))
env.fetch_info()
schedule = env.schedule
cal_dict = env.cal_dict
cal_names = env.cal_names

project_priorities = {
    4: 10,  
    5: 7
}

new_schedule = schedule_projects_daily(schedule, project_priorities, allow_multiday=True)
write_to_calendar(schedule, new_schedule, cal_dict, cal_names)
```

## Output

Directly onto the Google Calendar.

## License
MIT License. See `LICENSE` for details.


# To do:
 - PPO for basic schedule

# Project Updates:

### July 22, 2025

I got the actor and critic models working. But I have a feeling they are getting stuck in a local loss minimum so I added a "spiking" functionality. Essentially slightly modifies the parameters by multiplying in some noise and hopefully the model "escapes" the local minima and explores more.


### August 17, 2025.

I got an ILP solution, and got it working with the Google Calendar Interface. Looking goooood. Right now, I need to figure out the next stage, aka, ML models and how to get the compute for that.

