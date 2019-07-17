import torch

def FeatureL2Norm(feature):
    """
    Compute L2 normalized feature by channel-wise
    """
    epsilon = 1e-6

    # torch.pow takes the power of each element in input with exponent, exponent can be either a single float number
    # or a Tensor with the same number of elements as input
    # torch.sum reduce the dim 1, then use unsqueeze to add the dim 1 with size 1
    # expand_as enlarge the current tensor as original
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    # torch.div: each element of the tensor input is divided by each element of the tensor other
    # Return feature.shape: (batch_size, channels, h, w), such as (15, 15)
    return torch.div(feature, norm)