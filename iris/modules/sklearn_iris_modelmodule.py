from typing import Any, Dict, List

import numpy as np
from pathlib import Path
import pyrootutils

root = pyrootutils.setup_root(search_from=Path(__file__), indicator=[".project-root"], pythonpath=True)
from safetycage.modelmodule import ModelModule


class SklearnIrisModelModule(ModelModule):
    """Concrete ModelModule for sklearn classifiers on iris."""

    def __init__(
        self,
        selected_layers: List[str],
        use_onehot_encoder: bool,
        model: Any,
        **kwargs,
    ):
        super(SklearnIrisModelModule, self).__init__(
            selected_layers,
            use_onehot_encoder,
            model,
            **kwargs,
        )

        self.available_layers = {"input", "probabilities", "log_probabilities"}
        invalid_layers = [layer for layer in self.selected_layers if layer not in self.available_layers]
        if invalid_layers:
            raise ValueError(
                f"Unsupported selected_layers: {invalid_layers}. "
                f"Choose from {sorted(self.available_layers)}."
            )

        self.model_shape = self._calc_model_shape()
        self.last_layer = kwargs.get("last_layer", self.selected_layers[-1])

    def _ensure_2d(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x

    def _get_probabilities(self, x: np.ndarray) -> np.ndarray:
        x = self._ensure_2d(x)
        probabilities = self.model.predict_proba(x)
        return np.asarray(probabilities, dtype=np.float64)

    def _get_predictions(self, x: np.ndarray) -> np.ndarray:
        x = self._ensure_2d(x)
        predicted_classes = np.asarray(self.model.predict(x), dtype=np.int64)

        if self.use_onehot_encoder:
            num_classes = len(self.model.classes_)
            return np.eye(num_classes, dtype=np.float64)[predicted_classes]

        return predicted_classes

    def _get_activations(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        x = self._ensure_2d(x)
        probabilities = self._get_probabilities(x)
        log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))

        layer_map = {
            "input": x,
            "probabilities": probabilities,
            "log_probabilities": log_probabilities,
        }

        return {layer_name: layer_map[layer_name] for layer_name in self.selected_layers}

    def _get_pre_activations(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        # Tree ensembles do not have neural-network-style pre-activations.
        # We reuse the same feature representation for compatibility with SafetyCage.
        return self._get_activations(x)

    def _calc_model_shape(self) -> Dict[str, int]:
        n_features = int(getattr(self.model, "n_features_in_", 4))
        n_classes = int(len(getattr(self.model, "classes_", [0, 1, 2])))

        return {
            "input": n_features,
            "probabilities": n_classes,
            "log_probabilities": n_classes,
        }
