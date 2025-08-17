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
