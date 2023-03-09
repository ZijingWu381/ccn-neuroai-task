import torch
import torch.nn as nn

## Model metamers and adversarial examples
def net_extraxtor(model):
    """
    Extract subnetworks from a model

    # Arguments
        model: a pytorch model
    
    # Returns
        subnets: a dictionary of subnetworks
    """
    subnets = {}
    stack_of_layers = []
    for name, module in model.named_children():
        stack_of_layers.append(module)
        subnets[name] = nn.Sequential(*stack_of_layers)

    return subnets

def generate_random_image(size):
    """
    Generate random noise from Uniform(0, 1) with the specified size
    """
    image = torch.rand(size)
    image = image
    return image