import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from src.data.dataset import TabularDataset, ImagesDataset
from src.model.models import RNNRegressor, MultiCNNGRU
import mlflow
import mlflow.pytorch


class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: str = "cuda",
        mlflow_uri: str = "http://localhost:5000",
        experiment_name: str = "RNN_Regression",
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.experiment_name = experiment_name

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.history = {
            "train_r2": [],
            "val_r2": [],
            "train_rmse": [],
            "val_rmse": [],
        }

        # Set up MLflow
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(self.experiment_name)

    def train_epoch(self) -> tuple[float, float]:
        self.model.train()
        train_preds, train_targets = [], []

        for X_batch, y_batch in tqdm(
            self.train_loader, desc="Training", leave=False
        ):
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs.squeeze(), y_batch)

            loss.backward()
            self.optimizer.step()

            train_preds.extend(outputs.detach().cpu().numpy().flatten())
            train_targets.extend(y_batch.cpu().numpy().flatten())

        rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
        r2 = r2_score(train_targets, train_preds)

        return r2, rmse

    def validate(self) -> tuple[float, float]:
        self.model.eval()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = (
                    X_batch.to(self.device),
                    y_batch.to(self.device),
                )
                outputs = self.model(X_batch)
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_targets.extend(y_batch.cpu().numpy().flatten())

        rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        r2 = r2_score(val_targets, val_preds)

        return r2, rmse

    def run_training(
        self, num_epochs: int = 100, patience: int = 10
    ) -> tuple[nn.Module, dict[str, list[float]]]:
        best_val_r2 = -float("inf")
        epochs_without_improvement = 0
        best_model_weights = None

        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params(
                {
                    "NN Architecture": str(self.model),
                    "batch_size": self.batch_size,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "model_type": self.model.__class__.__name__,
                    "optimizer": self.optimizer.__class__.__name__,
                    "loss_function": self.criterion.__class__.__name__,
                    "device": self.device,
                    "num_epochs": num_epochs,
                    "patience": patience,
                }
            )

            for epoch in range(num_epochs):
                train_r2, train_rmse = self.train_epoch()
                val_r2, val_rmse = self.validate()

                # Update history
                self.history["train_r2"].append(train_r2)
                self.history["val_r2"].append(val_r2)
                self.history["train_rmse"].append(train_rmse)
                self.history["val_rmse"].append(val_rmse)

                # Log metrics
                mlflow.log_metrics(
                    {
                        "train_r2": train_r2,
                        "val_r2": val_r2,
                        "train_rmse": train_rmse,
                        "val_rmse": val_rmse,
                    },
                    step=epoch,
                )

                # Log progress
                print(f"Epoch {epoch + 1}/{num_epochs}:")
                print(
                    f"  Train R²: {train_r2:.4f}, Train RMSE: {train_rmse:.4f}"
                )
                print(f"  Val R²: {val_r2:.4f}, Val RMSE: {val_rmse:.4f}")

                # Early stopping check (now based on R²)
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    best_val_rmse = val_rmse
                    best_model_weights = self.model.state_dict()
                    epochs_without_improvement = 0

                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"\nEarly stopping at epoch {epoch + 1}!")
                        break

            # Load best model weights
            if best_model_weights is not None:
                self.model.load_state_dict(best_model_weights)
                # Log the final model
                # mlflow.pytorch.log_model(self.model, "best_model")
                mlflow.log_metric("best_val_r2", best_val_r2)
                mlflow.log_metric("best_val_rmse", best_val_rmse)

        return self.model, self.history


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def main():
    # Configuration
    config = {
        "seed": 42,
        "batch_size": 4,
        "num_epochs": 1000,
        "learning_rate": 0.001,
        "patience": 10,
        "data_path": Path("data/processed"),
        "model_type": "MultiCNNGRU",
        "hidden_size": 500,
        "num_layers": 2,
        "dropout": 0.3,
        "mlflow_uri": "",
        "experiment_name": "Test",
    }

    set_random_seeds(config["seed"])

    # Initialize datasets
    train_dataset = ImagesDataset(
        config["data_path"] / "X_train.csv",
        config["data_path"] / "y_train.csv",
    )
    test_dataset = ImagesDataset(
        config["data_path"] / "X_test.csv", config["data_path"] / "y_test.csv"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model
    # model = RNNRegressor(
    #     rnn_type=config["rnn_type"],
    #     input_size=train_dataset.X.shape[2],
    #     hidden_size=config["hidden_size"],
    #     num_layers=config["num_layers"],
    #     dropout=config["dropout"],
    #     device=device,
    # )
    model = MultiCNNGRU(
        num_frames=train_dataset.X.shape[1],
        hidden_size=config["hidden_size"],
        num_layers=1,
    )
    print(f"Training {config['model_type']} model on {device}")

    # Train model
    trainer = ModelTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        device=device,
        mlflow_uri=config["mlflow_uri"],
        experiment_name=config["experiment_name"],
    )

    trained_model, history = trainer.run_training(
        num_epochs=config["num_epochs"],
        patience=config["patience"],
    )


if __name__ == "__main__":
    main()
