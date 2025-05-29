import os
import subprocess
import platform
import psutil
from datetime import datetime
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from src.data.dataset import TabularDataset, ImagesDataset
from src.model.models import RNNRegressor, MultiCNNGRU
from tqdm import tqdm
from pathlib import Path
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv

PATH_PROCESSED = Path("data/processed")
PATH_MODELS = Path("models")


class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        random_state: int = 42,
        device: str = "cuda",
        mlflow_uri: str = "http://localhost:5000",
        experiment_name: str = "Test",
    ):
        """Инициализация тренера модели.

        Args:
            model: Модель PyTorch для обучения
            train_dataset: Датасет для обучения
            val_dataset: Датасет для валидации
            batch_size: Размер батча
            learning_rate: Скорость обучения
            random_state: Seed для воспроизводимости
            device: Устройство для обучения (cuda/cpu)
            mlflow_uri: URI для MLflow
            experiment_name: Имя эксперимента
        """
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

        self.random_state = random_state
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.history = {
            "train_r2": [],
            "val_r2": [],
            "train_rmse": [],
            "val_rmse": [],
        }

        # Set up MLflow
        if mlflow_uri:
            MLflowTracing.set_mlflow_s3()
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
        """Запуск процесса обучения.

        Args:
            num_epochs: Максимальное количество эпох
            patience: Количество эпох без улучшения для ранней остановки

        Returns:
            Обученная модель и история метрик
        """
        self.set_random_seeds(self.random_state)
        best_val_r2 = -float("inf")
        epochs_without_improvement = 0
        best_model_weights = None

        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params(
                {
                    # Системная информация
                    "system.commit_hash": MLflowTracing.get_git_commit_hash(),
                    "system.device": self.device,
                    **MLflowTracing.get_system_info(),
                    # Гиперпараметры
                    "hyperparams.random_state": self.random_state,
                    "hyperparams.batch_size": self.batch_size,
                    "hyperparams.learning_rate": self.optimizer.param_groups[
                        0
                    ]["lr"],
                    # Архитектура
                    "model.type": self.model.__class__.__name__,
                    "model.optimizer": self.optimizer.__class__.__name__,
                    "model.loss_function": self.criterion.__class__.__name__,
                    # Доп параметры модели
                    "model.architecture": str(self.model),
                    "model.num_params": sum(
                        p.numel() for p in self.model.parameters()
                    ),
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
                    best_epoch = epoch
                    best_model_weights = self.model.state_dict()
                    epochs_without_improvement = 0

                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"\nEarly stopping at epoch {epoch + 1}!")
                        break

            # Load best model weights
            if best_model_weights is not None:
                print(
                    f"Best model at epoch {best_epoch + 1} with R²: {best_val_r2:.4f}"
                )
                self.model.load_state_dict(best_model_weights)
                mlflow.log_metrics(
                    {
                        "best_val_r2": best_val_r2,
                        "best_val_rmse": best_val_rmse,
                    }
                )
                mlflow.pytorch.log_model(self.model, "best_model")
                if not PATH_MODELS.exists():
                    PATH_MODELS.mkdir()
                model_path = PATH_MODELS / (
                    f"{self.model.__class__.__name__}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M')}_"
                    f"r2_{val_r2:.4f}.pth"
                )
                torch.save(
                    self.model.state_dict(),
                    model_path,
                )
                print(f"Model saved to {model_path}")

        return self.model, self.history

    def set_random_seeds(self, seed: int = 42) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)


class MLflowTracing:
    """Класс для сбора системной информации и логирования в MLflow."""

    @staticmethod
    def set_mlflow_s3():
        load_dotenv()
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("S3_ACCESS_KEY_ID")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("S3_SECRET_ACCESS_KEY")
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = (
            "https://storage.yandexcloud.net"
        )

    @staticmethod
    def get_system_info() -> dict[str, str | float]:
        """Сбор информации о системе."""
        try:
            gpu_name = (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "None"
            )
            gpu_memory = (
                torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if torch.cuda.is_available()
                else 0
            )

            return {
                "system.system": platform.system(),
                "system.processor": platform.processor(),
                "system.cpu_cores_physical": psutil.cpu_count(logical=False),
                "system.cpu_cores_logical": psutil.cpu_count(logical=True),
                "system.total_memory_gb": round(
                    psutil.virtual_memory().total / (1024.0**3), 2
                ),
                "system.gpu_name": gpu_name,
                "system.gpu_memory_gb": round(gpu_memory, 2),
                "system.python_version": platform.python_version(),
                "system.torch_version": torch.__version__,
            }
        except Exception as e:
            print(f"Failed to get system info: {e}")
            return {}

    @staticmethod
    def get_git_commit_hash() -> str:
        """Получение хеша текущего коммита git."""
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("ascii")
                .strip()
            )
        except Exception as e:
            print(f"Failed to get git commit hash: {e}")
            return "unknown"


class Runner:
    @staticmethod
    def run_multicnngru():
        train_dataset = ImagesDataset(
            PATH_PROCESSED / "X_train.csv",
            PATH_PROCESSED / "y_train.csv",
        )
        test_dataset = ImagesDataset(
            PATH_PROCESSED / "X_test.csv",
            PATH_PROCESSED / "y_test.csv",
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = MultiCNNGRU(
            num_frames=train_dataset.X.shape[1],
            hidden_size=200,
            num_layers=1,
        )

        print(f"Training {model.__class__.__name__} model on {device}")

        # Train model
        trainer = ModelTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            batch_size=16,
            learning_rate=0.001,
            random_state=42,
            device=device,
            mlflow_uri="http://localhost:5000",
            experiment_name="Test",
        )

        trainer.run_training(
            num_epochs=1000,
            patience=10,
        )

    @staticmethod
    def run_rnn():
        train_dataset = TabularDataset(
            PATH_PROCESSED / "X_train.csv",
            PATH_PROCESSED / "y_train.csv",
        )
        test_dataset = TabularDataset(
            PATH_PROCESSED / "X_test.csv",
            PATH_PROCESSED / "y_test.csv",
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize model
        model = RNNRegressor(
            rnn_type="GRU",
            input_size=train_dataset.X.shape[2],
            hidden_size=200,
            num_layers=2,
            dropout=0.3,
            device=device,
        )

        print(f"Training {model.rnn_type} model on {device}")

        # Train model
        trainer = ModelTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            batch_size=128,
            learning_rate=0.001,
            random_state=42,
            device=device,
            mlflow_uri="http://localhost:5000",
            experiment_name="Test",
        )

        trainer.run_training(
            num_epochs=1000,
            patience=10,
        )


if __name__ == "__main__":
    Runner.run_rnn()
