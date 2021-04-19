import torch


#dummy function. replace with the actual neural network generator
def generate_noise_vector(in_vec, vec_size, device):
    return torch.randn(vec_size, device = device)
