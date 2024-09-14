import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, size=256):
        super(FiLM, self).__init__()
        self.linear1 = nn.Linear(size, size)
        self.linear2 = nn.Linear(size, size)

    def forward(self, x, aux):
        x = x * self.linear1(aux) + self.linear2(aux)
        return x
