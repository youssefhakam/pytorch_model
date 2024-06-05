import torch
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self):
        # Initialize the data here with appropriate shape
        self.x = torch.randn(100, 256)  # 100 samples, each with 256 features
        self.y = torch.randint(0, 10, (100,))  # 100 labels (class 0 to 9)

    def __len__(self):
        # Return the total number of samples
        return len(self.x)

    def __getitem__(self, idx):
        # Retrieve the sample at index idx
        return self.x[idx], self.y[idx]
