import torch
import numpy as np
from src.model.rnn import RNNRegressor
from src.data.dataset import Data
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    patience: int = 10,
    device: str = "cuda",
) -> tuple[nn.Module, dict[str, list[float]]]:
    """
    Train the GRU regression model with early stopping.

    Args:
        model: GRU model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for training
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate for optimizer
        patience: Number of epochs to wait before early stopping
        device: Device to use for training ('cuda' or 'cpu')

    Returns:
        tuple: (best_model, history) where history contains training and validation metrics
    """
    # Move model to device
    model = model.to(device)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize variables for early stopping
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_weights = None

    # History tracking
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_rmse": [],
        "val_rmse": [],
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        # Training loop
        for X_batch, y_batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"
        ):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss and metrics
            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy().flatten())
            train_targets.extend(y_batch.cpu().numpy().flatten())

        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_rmse = np.sqrt(
            np.mean((np.array(train_preds) - np.array(train_targets)) ** 2)
        )

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)

                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_targets.extend(y_batch.cpu().numpy().flatten())

        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_rmse = np.sqrt(
            np.mean((np.array(val_preds) - np.array(val_targets)) ** 2)
        )

        # Update history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)

        # Print progress
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(
            f"  Train Loss: {avg_train_loss:.4f}, Train RMSE: {train_rmse:.4f}"
        )
        print(f"  Val Loss: {avg_val_loss:.4f}, Val RMSE: {val_rmse:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}!")
                break

    # Load best model weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    return model, history


if __name__ == "__main__":
    seed = 42

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # NumPy
    np.random.seed(seed)

    PATH_PROCESSED = Path("data/processed")
    train_dataset = Data(
        PATH_PROCESSED / "X_train.csv", PATH_PROCESSED / "y_train.csv"
    )
    test_dataset = Data(
        PATH_PROCESSED / "X_test.csv", PATH_PROCESSED / "y_test.csv"
    )
    # Пример использования
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Создаем модель
    rnn_type = "GRU"
    model = RNNRegressor(
        rnn_type=rnn_type,
        input_size=train_dataset.X.shape[2],  # Количество фичей
        hidden_size=200,
        num_layers=2,
        dropout=0.3,
        device=device,
    )

    print(f"RNN Type: {rnn_type}")
    # Обучаем модель
    trained_model, history = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        batch_size=128,
        num_epochs=500,
        learning_rate=0.001,
        patience=10,
        device=device,
    )
