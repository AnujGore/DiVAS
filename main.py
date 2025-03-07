# import numpy as np
# from genetic_algorithm import initialize_population, fitness_function, predict_solution, update_population
# import argparse
# from rich.progress import track

# start_time = 9
# end_time = 18

# hours_worked = end_time-start_time

# num_tasks = 3

# timeslots = 4*hours_worked
# hidden_dim = 200
# task_weights = [0.4, 0.3, 0.3]


# def run_GA(population, timeslots, num_tasks, task_weights, num_parents, mutation_rate, alpha, generations):
#     for _ in track(range(generations), description="Running for Generations"):
#         avg_fitness_list = []
#         for individual in population:
#             this_solution = predict_solution(individual, timeslots = timeslots, fixed_blocks = [(13, 15)], num_tasks = num_tasks)
#             this_fitness = fitness_function(this_solution, num_tasks = num_tasks, task_weights = task_weights)

#             individual.fitness = this_fitness
#             avg_fitness_list.append(this_fitness)

#         new_population = update_population(population, num_parents = num_parents, mutation_rate = mutation_rate, alpha = alpha)
#         population = new_population

#     avg_fitness = np.mean(avg_fitness_list)

#     return avg_fitness

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()

#     parser.add_argument("--pop_count", type = int)
    
#     parser.add_argument("--hidden_dims", type = int)
#     parser.add_argument("--mutation_rate", type = float)
#     parser.add_argument("--alpha", type = float)
#     parser.add_argument("--num_parents", type = int)

#     env_vars_class = parser.parse_args()

#     population = initialize_population(env_vars_class.pop_count, timeslots = timeslots, hidden_dims = env_vars_class.hidden_dims)

#     run_GA(population, timeslots, num_tasks, task_weights, env_vars_class.num_parents, env_vars_class.mutation_rate, env_vars_class.alpha)


# alpha = 0.8983781024855365
# gen_count = 100
# hidden_dims = 50
# mutation_rate = 0.3313611154103022
# parent_count = 2


import numpy as np
import torch

num_tasks = 3

task_ids = [i+1 for i in range(num_tasks)]

start_time = 9
end_time = 18

hours_worked = end_time-start_time
day_array = np.zeros(shape = (4*hours_worked))

#Blocking
day_array[13:15] = -1 #Lunch


task_importance = torch.tensor([8, 5, 3]).float() #Out of 10
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

print(day_array)
