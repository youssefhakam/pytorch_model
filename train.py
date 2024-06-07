import torch.nn as nn
import torch.optim as optim
import torch
class Training:
    def __init__(self, model: nn.Module, learning_rate: float, optimizer_type: str, criterion_type: str, epochs: int, **kwargs) -> None:
        """
        Initialize the training loop.

        Args:
            model (nn.Module): Your neural network model.
            learning_rate (float): Learning rate for the optimizer.
            optimizer_type (str): 'SGD', 'Adam', etc.
            criterion_type (str): 'BCE', 'CrossEntropyLoss', etc.
            epochs (int): Number of training epochs.
        """
        self.model = model  # Ensure model is stored in the instance
        self.optimizer_type = optimizer_type
        self.criterion_type = criterion_type
        self.optimizer = self._initialize_optimizer(model, learning_rate, **kwargs)
        self.criterion = self._initialize_criterion()
        self.epochs = epochs

    def _initialize_optimizer(self, model: nn.Module, learning_rate: float, **kwargs):
        """
        Initialize the optimizer based on the specified type.

        Args:
            model (nn.Module): Your neural network model.
            learning_rate (float): Learning rate for the optimizer.

        Returns:
            torch.optim.Optimizer: Initialized optimizer.
        """
        optimizers = {
            'SGD': optim.SGD,
            'Adam': optim.Adam,
            'RMSprop': optim.RMSprop,
            'Adagrad': optim.Adagrad,
            'Adadelta': optim.Adadelta,
            'Adamax': optim.Adamax,
            'NAdam': optim.NAdam,
            'SparseAdam': optim.SparseAdam,
            'LBFGS': optim.LBFGS,
            'ASGD': optim.ASGD,
        }
        if self.optimizer_type not in optimizers:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
        return optimizers[self.optimizer_type](model.parameters(), lr=learning_rate, **kwargs)

    def _initialize_criterion(self):
        """
        Initialize the loss function based on the specified criterion.

        Returns:
            torch.nn.modules.loss._Loss: Initialized loss function.
        """
        criterions = {
            'BCE': nn.BCELoss(),
            'MSE': nn.MSELoss(),
            'NLL': nn.NLLLoss(),
            'KLDiv': nn.KLDivLoss(),
            'CrossEntropy': nn.CrossEntropyLoss(),
            'BCEWithLogits': nn.BCEWithLogitsLoss()
        }
        if self.criterion_type not in criterions:
            raise ValueError(f"Unsupported criterion: {self.criterion_type}")
        return criterions[self.criterion_type]

    def train(self, train_loader, val_loader):
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

            # Evaluate on the validation set
            val_loss, val_accuracy = self.evaluate(val_loader)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    def evaluate(self, val_loader) :
        """
        Evaluate the model on the validation set.

        Args:
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.

        Returns:
            float: Average validation loss.
            float: Validation accuracy.
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad() :
            for inputs, labels in val_loader :
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = val_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy
