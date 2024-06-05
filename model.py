import torch
import torch.nn as nn
from train import Training

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

  def fit(self, dataloader, learning_rate: float, optimizer_type: str, criterion_type: str, epochs: int):
        trainer = Training(self, learning_rate, optimizer_type, criterion_type, epochs)
        trainer.train(dataloader)
