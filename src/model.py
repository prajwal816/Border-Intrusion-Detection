"""
AI Model Integration — Audio Classification Model
===================================================
Loads trained CNN model for real-time acoustic classification.
Synchronous loading — TFLite loads in ~0.03s after TF import (~3s).
"""

import os
import time
import logging
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)

CLASS_LABELS = ["footstep", "gunshot", "noise"]


class AudioClassifier:
    """Audio classifier with TFLite/Keras/fallback support."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.interpreter = None
        self.model_type = "fallback"
        self.input_shape = None
        self.class_labels = CLASS_LABELS
        self._inference_times = []

        if model_path and os.path.exists(model_path):
            logger.info(f"[MODEL] Loading from: {model_path}")
            self._load_model(model_path)
        else:
            logger.warning(f"[MODEL] No model at {model_path}, using fallback")

    def _load_model(self, path: str):
        try:
            if path.endswith(".tflite"):
                self._load_tflite(path)
            elif path.endswith(".h5") or path.endswith(".keras"):
                self._load_keras(path)
            else:
                logger.error(f"[MODEL] Unsupported format: {path}")
        except Exception as e:
            logger.error(f"[MODEL] Load failed: {e}", exc_info=True)
            self.model_type = "fallback"

    def _load_keras(self, path: str):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(path, compile=False)
        self.model_type = "keras"
        self.input_shape = self.model.input_shape
        logger.info(f"[MODEL] Keras loaded: {self.input_shape}")

    def _load_tflite(self, path: str):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        inp = self.interpreter.get_input_details()
        self.input_shape = tuple(inp[0]['shape'])
        self.model_type = "tflite"
        logger.info(f"[MODEL] TFLite loaded: {self.input_shape}")

    def predict(self, features: np.ndarray) -> Dict[str, float]:
        t0 = time.time()
        if self.model_type == "keras" and self.model is not None:
            probs = self.model.predict(features, verbose=0)[0]
        elif self.model_type == "tflite" and self.interpreter is not None:
            probs = self._predict_tflite(features)
        else:
            probs = self._predict_fallback(features)

        ms = (time.time() - t0) * 1000
        self._inference_times.append(ms)
        if len(self._inference_times) > 100:
            self._inference_times = self._inference_times[-50:]

        result = {}
        for i, label in enumerate(self.class_labels):
            result[label] = float(probs[i]) if i < len(probs) else 0.0
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        return result

    def _predict_tflite(self, features):
        inp = self.interpreter.get_input_details()
        out = self.interpreter.get_output_details()
        features = features.astype(inp[0]['dtype'])
        self.interpreter.set_tensor(inp[0]['index'], features)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(out[0]['index'])[0]

    def _predict_fallback(self, features):
        if features is not None and features.size > 0:
            energy = np.mean(np.abs(features))
            variance = np.var(features)
            mx = np.max(np.abs(features))
            if mx > 3.0 and variance > 2.0:
                base = np.array([0.10, 0.75, 0.15])
            elif energy > 0.5 and variance < 1.5:
                base = np.array([0.70, 0.10, 0.20])
            else:
                base = np.array([0.15, 0.05, 0.80])
            noise = np.random.dirichlet(np.ones(3) * 10) * 0.15
            probs = base + noise
            return probs / probs.sum()
        return np.array([0.33, 0.33, 0.34])

    def get_last_inference_time(self):
        return self._inference_times[-1] if self._inference_times else 0.0

    def get_avg_inference_time(self):
        return float(np.mean(self._inference_times)) if self._inference_times else 0.0

    def get_model_info(self) -> Dict:
        return {
            "model_type": self.model_type or "fallback",
            "model_path": self.model_path,
            "input_shape": str(self.input_shape),
            "classes": self.class_labels,
            "avg_inference_ms": round(self.get_avg_inference_time(), 2),
        }
