"""
AI Model Integration — Subprocess-based Inference
===================================================
Runs ONNX/TFLite model in a separate Python process
to avoid DLL loading issues inside Streamlit.
"""

import os
import sys
import json
import time
import logging
import subprocess
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)

CLASS_LABELS = ["footstep", "gunshot", "noise"]

_SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "inference_server.py")


class AudioClassifier:
    """Audio classifier using subprocess inference."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model_type = "fallback"
        self.input_shape = None
        self.class_labels = CLASS_LABELS
        self._inference_times = []
        self._process = None

        if model_path and os.path.exists(model_path):
            print(f"[MODEL] Starting inference subprocess for: {model_path}")
            self._start_server(model_path)
        else:
            print(f"[MODEL] No model at {model_path}, using fallback")

    def _start_server(self, path: str):
        try:
            self._process = subprocess.Popen(
                [sys.executable, _SERVER_SCRIPT, path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            # Wait for ready signal
            ready_line = self._process.stdout.readline().strip()
            if ready_line:
                ready = json.loads(ready_line)
                if ready.get("status") == "ready":
                    self.model_type = ready.get("model_type", "unknown")
                    print(f"[MODEL] ✅ Subprocess ready — model: {self.model_type}")
                    return
                elif ready.get("error"):
                    print(f"[MODEL] ❌ Server error: {ready['error']}")

            self.model_type = "fallback"
        except Exception as e:
            print(f"[MODEL] ❌ Failed to start subprocess: {e}")
            self._process = None
            self.model_type = "fallback"

    def predict(self, features: np.ndarray) -> Dict[str, float]:
        t0 = time.time()

        if self._process and self._process.poll() is None:
            probs = self._predict_subprocess(features)
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

    def _predict_subprocess(self, features):
        try:
            request = json.dumps({"features": features.tolist()})
            self._process.stdin.write(request + "\n")
            self._process.stdin.flush()
            response_line = self._process.stdout.readline().strip()
            if response_line:
                response = json.loads(response_line)
                if "probs" in response:
                    return np.array(response["probs"])
                elif "error" in response:
                    print(f"[MODEL] Inference error: {response['error']}")
        except Exception as e:
            print(f"[MODEL] Subprocess comm error: {e}")
        return self._predict_fallback(None)

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

    def __del__(self):
        if self._process and self._process.poll() is None:
            try:
                self._process.stdin.write("QUIT\n")
                self._process.stdin.flush()
                self._process.wait(timeout=3)
            except Exception:
                self._process.kill()
