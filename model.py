import torch
import torch.nn as nn


## Create Your Model (I give you a example) : 
class Your_Model(nn.Module):
  def __init__(self):
    super(Your_Model, self).__init__()
    self.model = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        nn.Softmax(dim=1)
    )
  def forward(self, x):
    return self.model(x) 
