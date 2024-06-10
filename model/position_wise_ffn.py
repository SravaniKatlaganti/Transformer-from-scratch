import torch.nn as nn
import torch.nn.functional as F

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, dff):
        """
        Initialize the PositionWiseFeedForward module.
        """
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)

    def forward(self, x):
        """
        Forward pass for position-wise feed-forward network.
        """
        return self.linear2(F.relu(self.linear1(x)))
