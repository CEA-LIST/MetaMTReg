# @copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# @author CEA-LIST/DIASI/SIALV/LVA <quentin.bouniot@cea.fr>
# @license CECILL

import torch.nn as nn

def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class PrototypicalNetwork(nn.Module):
    def __init__(self, in_channels, hidden_size=64):
        super(PrototypicalNetwork, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )

    def forward(self, inputs):
        embeddings = self.encoder(inputs.view(-1, *inputs.shape[2:]))
        return embeddings.view(*inputs.shape[:2], -1)