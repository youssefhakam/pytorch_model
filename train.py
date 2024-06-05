import torch.nn as nn
import torch.optim as optim

class Training:
    def __init__(self, model: nn.Module, learning_rate: float, optimizer_type: str, criterion_type: str, epochs: int) -> None:
        """
        Initialize the training loop.

        Args:
            model (nn.Module): Your neural network model.
            learning_rate (float): Learning rate for the optimizer.
            optimizer_type (str): 'SGD' or 'Adam'.
            criterion_type (str): 'BCE' or 'CrossEntropyLoss'.
            epochs (int): Number of training epochs.
        """
        self.model = model  # Ensure model is stored in the instance
        self.optimizer_type = optimizer_type
        self.criterion_type = criterion_type
        self.optimizer = self._initialize_optimizer(model, learning_rate)
        self.criterion = self._initialize_criterion()
        self.epochs = epochs

    def _initialize_optimizer(self, model: nn.Module, learning_rate: float):
        """
        Initialize the optimizer based on the specified type.

        Args:
            model (nn.Module): Your neural network model.
            learning_rate (float): Learning rate for the optimizer.

        Returns:
            torch.optim.Optimizer: Initialized optimizer.
        """
        if self.optimizer_type == 'SGD':
            return optim.SGD(model.parameters(), lr=learning_rate)
        elif self.optimizer_type == 'Adam':
            return optim.Adam(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

    def _initialize_criterion(self):
        """
        Initialize the loss function based on the specified criterion.

        Returns:
            torch.nn.modules.loss._Loss: Initialized loss function.
        """
        if self.criterion_type == 'BCE':
            return nn.BCELoss()
        elif self.criterion_type == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion: {self.criterion_type}")

    def train(self, train_loader):
        """
        Perform the training loop.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        """
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                # Forward pass
                outputs = self.model(inputs)
            
                # Compute the loss
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accumulate the loss
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
