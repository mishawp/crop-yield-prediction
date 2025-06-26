import torch
from pathlib import Path
from src.data.dataset import (
    TabularDataset,
    ImagesDataset,
    OneImageDataset,
    MultiModalDataset,
)
from src.model.models import (
    RNNRegressor,
    FlexibleResNetRegressor,
    MultiCNNGRU,
    ResNetRegressor,
    EfficientNetB0Regressor,
    EfficientNetB4Regressor,
    MultiModalModel,
)
from src.model.train import ModelTrainer, MultiModalTrainer

PATH_PROCESSED = Path("data/processed")
PATH_MODELS = Path("models")


class Runner:
    @staticmethod
    def run_MultiCNNGRU():
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
            hidden_size=512,
            num_layers=1,
        )

        print(f"Training {model.__class__.__name__} model on {device}")

        # Train model
        trainer = ModelTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            batch_size=4,
            learning_rate=0.001,
            random_state=42,
            device=device,
            mlflow_uri="http://localhost:5000",
            experiment_name="MultiCNNGRU",
        )

        trainer.run_training(
            num_epochs=1000,
            patience=15,
        )

    @staticmethod
    def run_RNNRegressor():
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
            hidden_size=500,
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
            batch_size=16,
            learning_rate=0.001,
            random_state=42,
            device=device,
            mlflow_uri="http://localhost:5000",
            experiment_name="states6March-August",
        )

        trainer.run_training(
            num_epochs=1000,
            patience=10,
        )

    @staticmethod
    def run_ResNetRegressor():
        train_dataset = OneImageDataset(
            PATH_PROCESSED / "X_train.csv",
            PATH_PROCESSED / "y_train.csv",
        )
        test_dataset = OneImageDataset(
            PATH_PROCESSED / "X_test.csv",
            PATH_PROCESSED / "y_test.csv",
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize model
        model = ResNetRegressor()

        print(f"Training {model.__class__.__name__} model on {device}")

        # Train model
        trainer = ModelTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            batch_size=64,
            learning_rate=0.001,
            random_state=42,
            device=device,
            mlflow_uri="http://localhost:5000",
            experiment_name="ResNetRegressor",
        )

        trainer.run_training(
            num_epochs=1,
            patience=1,
        )

    @staticmethod
    def run_FlexibleResNetRegressor():
        train_dataset = OneImageDataset(
            PATH_PROCESSED / "X_train.csv",
            PATH_PROCESSED / "y_train.csv",
        )
        test_dataset = OneImageDataset(
            PATH_PROCESSED / "X_test.csv",
            PATH_PROCESSED / "y_test.csv",
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize model
        model = FlexibleResNetRegressor()

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
            experiment_name="FlexibleResNetRegressor",
        )

        trainer.run_training(
            num_epochs=1000,
            patience=15,
        )

    @staticmethod
    def run_EfficientNetB0Regressor():
        train_dataset = OneImageDataset(
            PATH_PROCESSED / "X_train.csv",
            PATH_PROCESSED / "y_train.csv",
        )
        test_dataset = OneImageDataset(
            PATH_PROCESSED / "X_test.csv",
            PATH_PROCESSED / "y_test.csv",
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize model
        model = EfficientNetB0Regressor()

        print(f"Training {model.__class__.__name__} model on {device}")

        # Train model
        trainer = ModelTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            batch_size=32,
            learning_rate=0.001,
            random_state=42,
            device=device,
            mlflow_uri="http://localhost:5000",
            experiment_name="FlexibleResNetRegressor",
        )

        trainer.run_training(
            num_epochs=1000,
            patience=15,
        )

    @staticmethod
    def run_EfficientNetB4Regressor():
        train_dataset = OneImageDataset(
            PATH_PROCESSED / "X_train.csv",
            PATH_PROCESSED / "y_train.csv",
        )
        test_dataset = OneImageDataset(
            PATH_PROCESSED / "X_test.csv",
            PATH_PROCESSED / "y_test.csv",
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize model
        model = EfficientNetB4Regressor()

        print(f"Training {model.__class__.__name__} model on {device}")

        # Train model
        trainer = ModelTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            batch_size=4,
            learning_rate=0.001,
            random_state=42,
            device=device,
            mlflow_uri="http://localhost:5000",
            experiment_name="EfficientNetB4Regressor",
        )

        trainer.run_training(
            num_epochs=1000,
            patience=15,
        )

    @staticmethod
    def run_multimodalmodel():
        train_dataset = MultiModalDataset(
            PATH_PROCESSED / "X_train.csv",
            PATH_PROCESSED / "y_train.csv",
        )
        test_dataset = MultiModalDataset(
            PATH_PROCESSED / "X_test.csv",
            PATH_PROCESSED / "y_test.csv",
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = MultiModalModel(
            num_frames=train_dataset.X_image.shape[1],
            rnns_input_size=train_dataset.X_tabular.shape[3],
            rnns_hidden_size=train_dataset.X_tabular.shape[3] * 10,
            rnns_num_layers=2,
            rnns_dropout=0.3,
            rnns_num_last_frames=2,
            main_hidden_size=None,
            main_num_layers=2,
            main_dropout=0.3,
            device=device,
        )

        print(f"Training {model.__class__.__name__} model on {device}")

        # Train model
        trainer = MultiModalTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            batch_size=16,
            learning_rate=0.001,
            random_state=42,
            device=device,
            mlflow_uri="http://localhost:5000",
            experiment_name="MultiModal",
        )

        trainer.run_training(
            num_epochs=1000,
            patience=20,
        )


if __name__ == "__main__":
    Runner.run_MultiCNNGRU()
