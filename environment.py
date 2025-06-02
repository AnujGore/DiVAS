import numpy as np
from google_calendar_interface import fetch_data

class Scheduler:
    def __init__(self, days, num_tasks):
        self.schedule = np.zeros(shape = (24*4, days))
        self.num_tasks = num_tasks
    
    def fetch_info(self):
        self.schedule = fetch_data(self.schedule)
    
    def new_event(self, action, idx, day):
        self.schedule[idx, day] = action
        return self.schedule[:, day]