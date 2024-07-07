"""
If you use this code, please cite the following papers:
(1) Lu, Ming, et al. "Federated Learning for Computational Pathology on Gigapixel Whole Slide Images." Medical Image Analysis. 2021.

The following code is adapted from: https://github.com/mahmoodlab/HistoFL, our code is based on this improvement.
"""

import torch
import pdb
import torch.nn.functional as F
from torch.distributions import Laplace


def sync_models(server_model, worker_models):
    server_params = server_model.state_dict()
    worker_models = [worker_model.load_state_dict(server_params) for worker_model in worker_models]

prev_global_params = None
def federated_averaging(model, worker_models, best_model_index, all_val_loss, weights=None , args=None):
    global prev_global_params
    if model.device is not None:
        central_device = model.device
    else:
        central_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    num_insti = len(worker_models)
    if weights is None:
        weights = [1/num_insti for i in range(num_insti)]  
    else:
        assert len(weights) == num_insti, "The length of weights does not match the number of workers"
        
    central_params = model.state_dict()
    all_worker_params = [worker_models[idx].state_dict() for idx in range(num_insti)]
    all_worker_params_noise = add_noise(all_worker_params,central_device,weights,args.noise_level)
    similarity_matrix = compute_similarity_matrix(all_worker_params_noise)
    all_worker_params_update = update_other_models(all_worker_params_noise, similarity_matrix, best_model_index,all_val_loss)
    similarity_matrix_1 = compute_similarity_matrix(all_worker_params_update)
    print(similarity_matrix_1)
    keys = central_params.keys()
    for key in keys:
        if 'labels' in key:
            continue
        else:
            temp = torch.zeros_like(central_params[key])  
            for idx in range(num_insti):
                temp = temp + weights[idx]*all_worker_params_update[idx][key].to(central_device)
            if prev_global_params is not None:
                central_params[key] = 1.0*temp + 0.0*prev_global_params[key]        
    model.load_state_dict(central_params)
    prev_global_params = central_params
    return model, worker_models


def add_noise(all_worker_params, central_device, weights,noise_level):
    num_insti = len(all_worker_params)
    for idx in range(num_insti):
        keys = all_worker_params[idx].keys()
        for key in keys:
            if 'labels' in key:
                continue
            elif noise_level > 0 and 'bias' not in key:  
                    laplace_dist = Laplace(torch.tensor([0.0]), torch.tensor([0.1))
                    noise = noise_level * laplace_dist.sample(all_worker_params[idx][key].size()).squeeze()
                    all_worker_params[idx][key] = all_worker_params[idx][key].to(central_device)+ noise.to(central_device)
                
            else:
                    all_worker_params[idx][key] = all_worker_params[idx][key].to(central_device)
   
    return all_worker_params

 
def compute_similarity_matrix(all_worker_params_noise):
    num_workers = len(all_worker_params_noise)
    similarity_matrix = torch.zeros((num_workers, num_workers)) 
    for i in range(num_workers):
        for j in range(i+1, num_workers):
            sim = 0
            count = 0
            for key in all_worker_params_noise[i].keys():
                if 'labels' in key:
                    continue
                elif 'bias' not in key:               
                    sim += F.cosine_similarity(all_worker_params_noise[i][key].reshape(-1), all_worker_params_noise[j][key].reshape(-1), dim=0)
                    count +=1

            sim /= count
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim
    
    return similarity_matrix

previous_loss = None 
alpha =  0.5
max_epochs = 30

def update_other_models(all_worker_params_noise, similarity_matrix, best_model_index, current_loss):
    num_insti = len(all_worker_params_noise)
    alpha_factor = 1.0
    global alpha
    global previous_loss 
    global max_epochs

    for idx in range(num_insti):
        if idx == best_model_index:
            continue
        else:
            for key in all_worker_params_noise[idx].keys():
                if 'labels' in key:
                    continue
                elif 'bias' not in key:
                    delta_W_local = (all_worker_params_noise[best_model_index][key] -all_worker_params_noise[idx][key] ) * alpha * similarity_matrix[best_model_index][idx]
                    all_worker_params_noise[idx][key] += delta_W_local
            # all_worker_params_noise[idx] = 0.99*all_worker_params_noise[idx] + 0.01*all_worker_params_noise[best_model_index]   
    if max_epochs > 0:
        if previous_loss is not None:
            if current_loss <= previous_loss:
                alpha_factor *= 1.1  # Increase alpha gradually if the loss improves
            else:
                alpha_factor *= 0.8  # Decrease alpha if the loss worsens
    else:
        alpha = 0.5
        max_epochs = 30
   
    max_epochs -= 1

    print("max_epochs:",max_epochs)
    print("Best Model Index:", best_model_index)
    print("previous_loss",previous_loss)
    print("current_loss",current_loss)

    alpha *= alpha_factor
    previous_loss = current_loss
    print("Alpha:", alpha)

    return all_worker_params_noise
