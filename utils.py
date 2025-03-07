import torch

def apply_fixed_blocks(chromosome, blocks):
    for block in blocks:
        chromosome[block[0]:block[1]] = 0
    return chromosome

def assign_timeslots(chromosome, **kwargs):
    bins = torch.linspace(min(chromosome), max(chromosome), kwargs["num_tasks"])
    discretized = torch.bucketize(chromosome, bins, right=True)
    return torch.Tensor(discretized)