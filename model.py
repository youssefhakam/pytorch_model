import torch
import torch.nn as nn
from train import Training
from dataset import SimpleDataset

## Create Your Model (I give you a example) : 
class YourModel(nn.Module):
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
    """
    Fits the neural network model to the training data.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader containing training data.
        learning_rate (float): Learning rate for the optimizer.
        optimizer_type (str): Type of optimizer ('SGD' or 'Adam').
        criterion_type (str): Type of loss function ('BCE' or 'CrossEntropyLoss').
        epochs (int): Number of training epochs.
    """
    trainer = Training(self, learning_rate, optimizer_type, criterion_type, epochs)
    trainer.train(dataloader)


dataset = SimpleDataset()
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
model = YourModel()
model.fit(dataloader, 0.01, 'SGD', 'CrossEntropyLoss', 100)
