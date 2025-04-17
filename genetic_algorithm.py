import torch
from divas import divas

from utils import apply_fixed_blocks, assign_timeslots

def initialize_population(pop_size, **kwargs):
    """
    kwargs = timeslots, hidden_dims
    """
    single_chromosome_size = kwargs["timeslots"]
    hidden_dimensions = kwargs["hidden_dims"]

    models = [divas(single_chromosome_size, hidden_dimensions, device = "cpu") for _ in range(pop_size)]

    return models

def predict_solution(individual: divas, **kwargs):
    """
    kwargs = timeslots, fixed_blocks, num_tasks
    """

    single_structure = torch.ones(size = (kwargs["timeslots"], ))
    single_structure = individual.predict(single_structure)
    single_structure = assign_timeslots(single_structure, num_tasks = kwargs["num_tasks"])
    
    return apply_fixed_blocks(single_structure, blocks=kwargs["fixed_blocks"]) 
        

def fitness_function(single_chromosome, **kwargs):
    """
    kwargs = num_tasks, task_weights
    """
    this_score = 0

    #Criteria 1 (correct distribution)
    chromosome_task_weight = torch.zeros(size = (kwargs["num_tasks"], ))
    for i, task_id in enumerate(range(1, kwargs["num_tasks"]+1)):
        chromosome_task_weight[i] = torch.count_nonzero((single_chromosome == task_id).to(int))

    chromosome_task_weight = torch.nn.functional.normalize(chromosome_task_weight, p=1.0, dim = 0)

    this_score -= torch.sum(torch.abs(chromosome_task_weight - torch.Tensor(kwargs["task_weights"])))

    #Criteria 2 (long blocks)
    chromosome_task_length = torch.zeros(size = (kwargs["num_tasks"], ))
    for j, task_id in enumerate(range(1, kwargs["num_tasks"]+1)):
        idx = (single_chromosome == task_id).to(int)
        max_length = 0; this_length = 0
        for s in idx:
            if s == 1:
                this_length += 1
            else:
                if this_length > max_length:
                    max_length = this_length
                this_length = 0
        
        chromosome_task_length[j] = max_length

    chromosome_task_length = torch.nn.functional.normalize(chromosome_task_length, p=1.0, dim = 0)
    this_score -= torch.sum(torch.abs(chromosome_task_length - torch.Tensor(kwargs["task_weights"])))

    return this_score.item()


def crossover(model1: divas, model2: divas, alpha):
    if alpha == "self":
        alpha = model1.fitness/model2.fitness
    else:
        alpha = alpha

    child = divas(model1.input_dim, model1.hidden_dim, device = model1.device)

    child_layer1 = torch.clip(torch.mul(alpha, model1.input_to_hidden.clone()) + torch.mul((1 - alpha), model2.input_to_hidden.clone()), min = -1.0, max = 1.0)
    child_layer2 = torch.clip(torch.mul(alpha, model1.first_hidden.clone()) + torch.mul((1 - alpha), model2.first_hidden.clone()), min = -1.0, max = 1.0)
    child_layer3 = torch.clip(torch.mul(alpha, model1.second_hidden.clone()) + torch.mul((1 - alpha), model2.second_hidden.clone()), min = -1.0, max = 1.0)
    child_layer4 = torch.clip(torch.mul(alpha, model1.hidden_to_output.clone()) + torch.mul((1 - alpha), model2.hidden_to_output.clone()), min = -1.0, max = 1.0)

    child.input_to_hidden = child_layer1
    child.first_hidden = child_layer2
    child.second_hidden = child_layer3
    child.hidden_to_output = child_layer4

    return child


def sort_population(population: list[divas]):
    population_fitness = []

    for ind in population:
        population_fitness.append(ind.fitness)
    
    best_ranked_population = [x for _, x in sorted(zip(torch.argsort(torch.Tensor(population_fitness)), population))]

    return best_ranked_population


def roulette_wheel_selection(population: list[divas]):
    population_fitness = []

    for ind in population:
        population_fitness.append(ind.fitness)


    population_fitness = torch.Tensor(population_fitness)

    population_fitness_as_p = torch.softmax(population_fitness, 0)

    parent_idx = torch.multinomial(population_fitness_as_p, 1)

    return parent_idx


def update_population(population, num_parents = 2, mutation_rate = 0.1, alpha = 0.5):
    fittest_pop = sort_population(population)

    #Step 1: Pick best individuals to f*ck
    new_population = fittest_pop[:num_parents]

    while len(new_population) < len(population):

        parent1 = roulette_wheel_selection(population)
        parent2 = roulette_wheel_selection(population)

        child = crossover(population[parent1], population[parent2], alpha)

        if torch.randn(1) < mutation_rate:
            child.mutate()

        new_population.append(child)

    return new_population


# # alpha = 0.8983781024855365
# # gen_count = 100
# # hidden_dims = 50
# # mutation_rate = 0.3313611154103022
# # parent_count = 2