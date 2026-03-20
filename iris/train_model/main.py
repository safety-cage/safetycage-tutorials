import joblib

import numpy as np
import pyrootutils
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
root_iris = root / "iris"
root_iris_metadata = root_iris / "metadata"
data_dir = root_iris_metadata / "data"
model_dir = root_iris_metadata / "model"

from iris.modules.sklearn_iris_datamodule import IrisDataModule


def decode_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim > 1:
        return np.argmax(y, axis=1)
    return y.astype(int)


def evaluate_split(name: str, model, x: np.ndarray, y: np.ndarray) -> None:
    y_true = decode_labels(y)
    y_pred = model.predict(x)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{name} accuracy: {accuracy:.4f}")
    print(f"{name} confusion matrix:\n{cm}")


def main():
    data_module_args = {
        "data_dir": data_dir,
        "from_cache": True,
        "batch_size": 32,
        "val_split": 0.2,
        "test_split": 0.2,
        "use_onehot_encoder": True,
        "standardize": True,
        "random_state": 42,
        "device": "cpu",
    }

    data_module = IrisDataModule(**data_module_args)

    x_train, y_train = data_module.data_train
    x_val, y_val = data_module.data_val
    x_test, y_test = data_module.data_test

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        max_depth=None,
        min_samples_leaf=1,
    )
    model.fit(x_train, decode_labels(y_train))

    evaluate_split("Train", model, x_train, y_train)
    evaluate_split("Validation", model, x_val, y_val)
    evaluate_split("Test", model, x_test, y_test)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "random_forest.joblib")

    data_module.to_joblib(data_dir / "iris_data_module.joblib")

    print("\nSaved model to:", model_dir / "random_forest.joblib")
    print("Saved data module to:", data_dir / "iris_data_module.joblib")


if __name__ == "__main__":
    main()
