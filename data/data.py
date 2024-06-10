import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(batch_size=8):
    """
    Load and prepare the dataset.
    For demonstration, we use a dummy dataset.
    """
    inputs = torch.randint(0, 1000, (64, 50))  # Random input data
    targets = torch.randint(0, 1000, (64, 50))  # Random target data
    dataset = TensorDataset(inputs, targets)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    return data_loader
