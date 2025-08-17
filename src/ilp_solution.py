from ortools.linear_solver import pywraplp
import numpy as np

def schedule_projects_daily(schedule, project_priorities, allow_multiday=True, switch_penalty=2.0):
    """
    Linear programming scheduler for DiVAS on a per-day basis with optional multi-day spillover
    and automatic reduction of unnecessary project switching.

    schedule: np.array shape (n_slots, n_days)
              -> each cell = 0 if free, nonzero if busy
    project_priorities: dict {project_id: priority (1-10)}
    allow_multiday: bool, whether to allow projects to continue to next day if not fully scheduled
    switch_penalty: float, weight to penalize switching projects in consecutive slots

    Returns:
        new_schedule: np.array same shape as input,
                      with free slots filled with project IDs
    """
    n_slots, n_days = schedule.shape
    proj_ids = list(project_priorities.keys())
    new_schedule = schedule.copy()
    
    # Track leftover slots for multi-day allocation
    leftover_slots = {p: 0 for p in proj_ids}

    for d in range(n_days):
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            raise Exception("Solver not available")

        x = {}
        for i in range(n_slots):
            for p in proj_ids:
                x[(i,p)] = solver.BoolVar(f"x_{i}_{p}")

        # Busy slots remain unchanged
        for i in range(n_slots):
            if schedule[i,d] != 0:
                for p in proj_ids:
                    solver.Add(x[(i,p)] == 0)

        # Exactly one project per free slot
        for i in range(n_slots):
            if schedule[i,d] == 0:
                solver.Add(sum(x[(i,p)] for p in proj_ids) == 1)
            else:
                solver.Add(sum(x[(i,p)] for p in proj_ids) == 0)

        # Proportional allocation + leftover slots
        total_free_slots = sum(schedule[i,d]==0 for i in range(n_slots))
        for p in proj_ids:
            min_slots = int(np.floor(total_free_slots * project_priorities[p] / sum(project_priorities.values())))
            if allow_multiday:
                min_slots += leftover_slots[p]
            solver.Add(sum(x[(i,p)] for i in range(n_slots)) >= min_slots)

        # Objective: maximize priorities while penalizing switches
        objective = solver.Objective()
        for i in range(n_slots):
            for p in proj_ids:
                objective.SetCoefficient(x[(i,p)], project_priorities[p])
                
        # Penalize switching between consecutive slots
        for i in range(n_slots-1):
            for p1 in proj_ids:
                for p2 in proj_ids:
                    if p1 != p2:
                        # Binary variable to detect a switch
                        switch = solver.BoolVar(f"switch_{i}_{p1}_{p2}")
                        solver.Add(switch >= x[(i,p1)] + x[(i+1,p2)] - 1)
                        objective.SetCoefficient(switch, -switch_penalty)

        objective.SetMaximization()

        # Solve
        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            raise Exception(f"No optimal solution found for day {d}")

        # Fill schedule and calculate leftover slots
        for p in proj_ids:
            assigned_slots = sum(x[(i,p)].solution_value() > 0.5 for i in range(n_slots))
            leftover_slots[p] = max(0, int(np.floor(total_free_slots * project_priorities[p] / sum(project_priorities.values()))) - assigned_slots)

        for i in range(n_slots):
            if schedule[i,d] == 0:
                for p in proj_ids:
                    if x[(i,p)].solution_value() > 0.5:
                        new_schedule[i,d] = p

    return new_schedule