"""
AI Model Integration — Audio Classification Model
===================================================
Loads trained CNN model for real-time acoustic classification.
Supports both Keras (.h5) and TFLite (.tflite) inference.
"""

import os
import time
import logging
import numpy as np
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Class labels — must match training order
CLASS_LABELS = ["footstep", "gunshot", "noise"]


class AudioClassifier:
    """
    Audio classification model for border intrusion detection.
    
    Supports:
    - Keras H5 model (full TensorFlow)
    - TFLite model (edge-optimized)
    - Fallback random classifier for testing
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.interpreter = None
        self.model_type = None  # "keras", "tflite", or "fallback"
        self.input_shape = None
        self.class_labels = CLASS_LABELS
        self._inference_times = []
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            logger.warning(f"[MODEL] Model not found at {model_path}, using fallback classifier")
            self.model_type = "fallback"
    
    def _load_model(self, path: str):
        """Load model from file."""
        try:
            if path.endswith(".tflite"):
                self._load_tflite(path)
            elif path.endswith(".h5") or path.endswith(".keras"):
                self._load_keras(path)
            else:
                logger.error(f"[MODEL] Unsupported model format: {path}")
                self.model_type = "fallback"
        except Exception as e:
            logger.error(f"[MODEL] Failed to load model: {e}")
            self.model_type = "fallback"
    
    def _load_keras(self, path: str):
        """Load Keras H5 model."""
        import tensorflow as tf
        self.model = tf.keras.models.load_model(path)
        self.model_type = "keras"
        self.input_shape = self.model.input_shape
        logger.info(f"[MODEL] Keras model loaded: {path}")
        logger.info(f"[MODEL] Input shape: {self.input_shape}")
        logger.info(f"[MODEL] Parameters: {self.model.count_params():,}")
    
    def _load_tflite(self, path: str):
        """Load TFLite model."""
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        self.input_shape = input_details[0]['shape']
        self.model_type = "tflite"
        logger.info(f"[MODEL] TFLite model loaded: {path}")
        logger.info(f"[MODEL] Input shape: {self.input_shape}")
    
    def predict(self, features: np.ndarray) -> Dict[str, float]:
        """
        Run inference on preprocessed MFCC features.
        
        Args:
            features: numpy array of shape (1, N_MFCC, time_steps, 1)
            
        Returns:
            dict mapping class labels to probabilities
        """
        start_time = time.time()
        
        if self.model_type == "keras":
            probabilities = self._predict_keras(features)
        elif self.model_type == "tflite":
            probabilities = self._predict_tflite(features)
        else:
            probabilities = self._predict_fallback(features)
        
        inference_time = (time.time() - start_time) * 1000  # ms
        self._inference_times.append(inference_time)
        if len(self._inference_times) > 100:
            self._inference_times = self._inference_times[-50:]
        
        # Build result dict
        result = {}
        for i, label in enumerate(self.class_labels):
            result[label] = float(probabilities[i]) if i < len(probabilities) else 0.0
        
        # Ensure probabilities sum to ~1.0
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        
        return result
    
    def _predict_keras(self, features: np.ndarray) -> np.ndarray:
        """Keras model inference."""
        predictions = self.model.predict(features, verbose=0)
        return predictions[0]
    
    def _predict_tflite(self, features: np.ndarray) -> np.ndarray:
        """TFLite model inference."""
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        # Ensure correct dtype
        input_dtype = input_details[0]['dtype']
        features = features.astype(input_dtype)
        
        self.interpreter.set_tensor(input_details[0]['index'], features)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(output_details[0]['index'])
        return predictions[0]
    
    def _predict_fallback(self, features: np.ndarray) -> np.ndarray:
        """
        Fallback classifier using simple energy-based heuristics.
        Used when no trained model is available.
        """
        # Use feature statistics as rough classification heuristic
        if features is not None and features.size > 0:
            energy = np.mean(np.abs(features))
            variance = np.var(features)
            max_val = np.max(np.abs(features))
            
            # High energy + high variance → likely gunshot
            # Moderate energy + rhythmic → likely footstep
            # Low energy → likely noise
            
            if max_val > 3.0 and variance > 2.0:
                base = np.array([0.10, 0.75, 0.15])
            elif energy > 0.5 and variance < 1.5:
                base = np.array([0.70, 0.10, 0.20])
            else:
                base = np.array([0.15, 0.05, 0.80])
            
            # Add slight randomness
            noise = np.random.dirichlet(np.ones(3) * 10) * 0.15
            probs = base + noise
            probs = probs / probs.sum()
            return probs
        
        return np.array([0.33, 0.33, 0.34])
    
    def get_last_inference_time(self) -> float:
        """Get last inference time in ms."""
        return self._inference_times[-1] if self._inference_times else 0.0
    
    def get_avg_inference_time(self) -> float:
        """Get average inference time in ms."""
        return np.mean(self._inference_times) if self._inference_times else 0.0
    
    def get_model_info(self) -> Dict:
        """Get model metadata."""
        return {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "input_shape": str(self.input_shape),
            "classes": self.class_labels,
            "avg_inference_ms": round(self.get_avg_inference_time(), 2),
        }
