from typing import Optional, Dict, Tuple

import joblib
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import pyrootutils
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

root = pyrootutils.setup_root(search_from=Path(__file__), indicator=[".project-root"], pythonpath=True)
from safetycage.datamodule import DataModule


class IrisDataModule(DataModule):
    """Concrete DataModule for the sklearn iris dataset."""

    def __init__(
        self,
        data_dir=None,
        from_cache: bool = True,
        batch_size: int = 32,
        val_split: float = 0.2,
        test_split: float = 0.9,
        use_onehot_encoder: bool = True,
        standardize: bool = True,
        random_state: int = 42,
        device: str = "cpu",
    ) -> None:
        super().__init__(data_dir, from_cache, batch_size, device)

        self.val_split = val_split
        self.test_split = test_split
        self.use_onehot_encoder = use_onehot_encoder
        self.standardize = standardize
        self.random_state = random_state
        self.scaler: Optional[StandardScaler] = None
        self.feature_names = [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]

        self.setup()

    @property
    def classes(self) -> Dict[int, str]:
        return {
            0: "setosa",
            1: "versicolor",
            2: "virginica",
        }

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def dataset_name(self) -> str:
        return "iris"

    def setup(self) -> None:
        path_data = (self.data_dir / self.dataset_name).with_suffix(".npz")
        x, y = self._load_data(path_data)

        x_train_val, y_train_val, x_test, y_test = self._split(
            x,
            y,
            split=self.test_split,
        )

        x_train, y_train, x_val, y_val = self._split(
            x=x_train_val,
            y=y_train_val,
            split=self.val_split,
        )

        x_train, y_train = self._transform(x_train, y_train, fit_scaler=True)
        x_val, y_val = self._transform(x_val, y_val, fit_scaler=False)
        x_test, y_test = self._transform(x_test, y_test, fit_scaler=False)

        self.data_train = (x_train, y_train)
        self.data_val = (x_val, y_val)
        self.data_test = (x_test, y_test)

    def _split(self, x, y, split):
        """Split the dataset into training and testing sets.

        Args:
            x (np.ndarray): The input features.
            y (np.ndarray): The input labels.
            split (float): The proportion of the dataset to include in the test split.

        Returns:
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: The training and testing data.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, stratify=y, random_state=42, test_size=split
        )
        
        return x_train, y_train, x_test, y_test

    def _load_data(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        if self.from_cache:
            try:
                data = np.load(filepath)
                print(f"Data loaded from {filepath}.")
                return data["x"], data["y"]
            except (FileNotFoundError, IOError):
                print(f"Data not found at {filepath}.\nDownloading from sklearn...")
                return self._download_data(filepath)

        print("Downloading iris data from sklearn...")
        return self._download_data(filepath)

    def _download_data(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset = load_iris()
        x = dataset.data.astype(np.float64)
        y = dataset.target.astype(np.int64)
        np.savez(path, x=x, y=y)
        return x, y

    def _transform(self, x, y, fit_scaler: bool = False):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).reshape(-1)

        if self.standardize:
            if fit_scaler or self.scaler is None:
                self.scaler = StandardScaler()
                x = self.scaler.fit_transform(x)
            else:
                x = self.scaler.transform(x)

        x = x.astype(np.float64)

        if self.use_onehot_encoder:
            y_encoded = np.eye(self.num_classes, dtype=np.float64)[y]
            return x, y_encoded

        return x, y

    def set_predictions(self, predictions: Dict[str, np.ndarray]) -> None:
        y_train_pred = predictions.get("y_pred_train")
        y_val_pred = predictions.get("y_pred_val")
        y_test_pred = predictions.get("y_pred_test")

        if y_train_pred is None or y_val_pred is None or y_test_pred is None:
            raise ValueError(
                "Predictions dictionary must contain 'y_pred_train', 'y_pred_val', and 'y_pred_test' keys"
            )

        self.data_train = (self.data_train[0], self.data_train[1], y_train_pred)
        self.data_val = (self.data_val[0], self.data_val[1], y_val_pred)
        self.data_test = (self.data_test[0], self.data_test[1], y_test_pred)

    def to_joblib(self, path: Optional[str] = None):
        if path is None:
            path = (self.data_dir / f"{self.dataset_name}_data_module").with_suffix(".joblib")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    def from_joblib(self, path: Optional[str] = None):
        if path is None:
            path = (self.data_dir / f"{self.dataset_name}_data_module").with_suffix(".joblib")

        return joblib.load(path)

    def plot_feature_pairs(self) -> None:
        x_train, y_train = self.data_train[:2]
        y_labels = np.argmax(y_train, axis=1) if self.use_onehot_encoder else y_train

        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        axes = axes.flatten()
        feature_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        for ax, (i, j) in zip(axes, feature_pairs):
            for class_id, class_name in self.classes.items():
                mask = y_labels == class_id
                ax.scatter(x_train[mask, i], x_train[mask, j], label=class_name, alpha=0.7)
            ax.set_xlabel(self.feature_names[i])
            ax.set_ylabel(self.feature_names[j])

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3)
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.show()
