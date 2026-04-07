"""
VirtualESP32Node — Hardware Abstraction Layer
==============================================
Simulates an ESP32 microcontroller with MEMS microphone
for edge AI acoustic surveillance at border zones.
"""

import time
import random
import threading
import logging
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class NodeState(Enum):
    """ESP32 node operating states."""
    BOOT = "BOOTING"
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    CAPTURING = "CAPTURING"
    PROCESSING = "PROCESSING"
    TRANSMITTING = "TRANSMITTING"
    SLEEP = "DEEP_SLEEP"
    ERROR = "ERROR"


class AlertLevel(Enum):
    """Security alert levels."""
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    INTRUSION = "INTRUSION"
    HIGH_ALERT = "HIGH_ALERT"
    CRITICAL = "CRITICAL"


@dataclass
class NodeTelemetry:
    """Node hardware telemetry data."""
    battery_percent: float = 100.0
    temperature_c: float = 25.0
    uptime_seconds: float = 0.0
    total_inferences: int = 0
    total_alerts: int = 0
    total_packets_sent: int = 0
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    last_wake_reason: str = "POWER_ON"


@dataclass
class InferenceResult:
    """Result from TinyML inference on edge."""
    predictions: Dict[str, float] = field(default_factory=dict)
    predicted_class: str = ""
    confidence: float = 0.0
    inference_time_ms: float = 0.0
    alert_level: AlertLevel = AlertLevel.NORMAL
    timestamp: str = ""
    node_id: str = ""


class VirtualESP32Node:
    """
    Simulates an ESP32 microcontroller node deployed at a border zone.
    
    Hardware simulation includes:
    - MEMS microphone audio sampling
    - Frame buffering (ring buffer)
    - Low-power wake-on-sound logic
    - TinyML inference engine
    - Power management & battery simulation
    - Watchdog timer
    """
    
    # Simulated hardware specs
    CPU_FREQ_MHZ = 240           # ESP32 dual-core
    FLASH_SIZE_MB = 4
    SRAM_KB = 520
    BATTERY_CAPACITY_MAH = 3000
    DRAIN_RATE_ACTIVE = 0.015    # % per second when active
    DRAIN_RATE_SLEEP = 0.001     # % per second in deep sleep
    
    def __init__(self, node_id: str = "01", position: tuple = (0.0, 0.0)):
        self.node_id = node_id
        self.position = position
        self.state = NodeState.BOOT
        self.alert_level = AlertLevel.NORMAL
        self.telemetry = NodeTelemetry()
        self._boot_time = time.time()
        self._last_update = time.time()
        self._state_log = []
        self._inference_history = []
        self._is_running = False
        self._lock = threading.Lock()
        
        # Simulated hardware registers
        self._wake_on_sound_enabled = True
        self._adc_buffer_size = 22050  # 1 second at 22050Hz
        self._watchdog_timeout = 30.0  # seconds
        self._last_watchdog_feed = time.time()
        
        self._log(f"ESP32 Node {self.node_id} powered on")
        self._log(f"CPU: {self.CPU_FREQ_MHZ}MHz | Flash: {self.FLASH_SIZE_MB}MB | SRAM: {self.SRAM_KB}KB")
    
    def _log(self, message: str):
        """Internal logging with ESP32-style format."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [ESP32 NODE {self.node_id}] {message}"
        self._state_log.append(log_entry)
        logger.info(log_entry)
        # Keep log bounded
        if len(self._state_log) > 500:
            self._state_log = self._state_log[-300:]
    
    def boot(self):
        """Simulate ESP32 boot sequence."""
        self._set_state(NodeState.BOOT)
        self._log("Initializing hardware...")
        self._log("MEMS Microphone: ICS-43434 initialized")
        self._log("ADC: 12-bit SAR configured @ 22050Hz")
        self._log(f"Wake-on-sound threshold: ENABLED")
        self._log("LoRa SX1276 radio: initialized (SF7, BW125)")
        self._log(f"TinyML model: loaded ({self.FLASH_SIZE_MB}MB flash)")
        self._log("Boot complete — entering IDLE state")
        self._set_state(NodeState.IDLE)
        self._is_running = True
        self.telemetry.last_wake_reason = "POWER_ON"
    
    def _set_state(self, new_state: NodeState):
        """Transition to a new state."""
        old_state = self.state
        self.state = new_state
        if old_state != new_state:
            self._log(f"State: {old_state.value} → {new_state.value}")
    
    def _update_telemetry(self):
        """Update simulated hardware telemetry."""
        now = time.time()
        elapsed = now - self._last_update
        self._last_update = now
        
        # Battery drain
        if self.state in (NodeState.CAPTURING, NodeState.PROCESSING, NodeState.TRANSMITTING):
            self.telemetry.battery_percent -= self.DRAIN_RATE_ACTIVE * elapsed
        elif self.state == NodeState.SLEEP:
            self.telemetry.battery_percent -= self.DRAIN_RATE_SLEEP * elapsed
        else:
            self.telemetry.battery_percent -= self.DRAIN_RATE_ACTIVE * 0.5 * elapsed
        
        self.telemetry.battery_percent = max(0.0, self.telemetry.battery_percent)
        
        # Uptime
        self.telemetry.uptime_seconds = now - self._boot_time
        
        # Temperature simulation (slight variation)
        base_temp = 25.0 if self.state == NodeState.SLEEP else 32.0
        self.telemetry.temperature_c = base_temp + random.uniform(-1.5, 3.0)
        
        # CPU/Memory simulation
        state_cpu = {
            NodeState.SLEEP: 2.0, NodeState.IDLE: 8.0, NodeState.LISTENING: 15.0,
            NodeState.CAPTURING: 35.0, NodeState.PROCESSING: 85.0, NodeState.TRANSMITTING: 45.0,
        }
        target_cpu = state_cpu.get(self.state, 10.0)
        self.telemetry.cpu_usage_percent = target_cpu + random.uniform(-5, 5)
        self.telemetry.memory_usage_percent = 40.0 + random.uniform(-5, 10)
    
    def capture_audio_frame(self):
        """Simulate MEMS microphone audio frame capture."""
        self._set_state(NodeState.CAPTURING)
        self._log("Capturing audio frame...")
        self._log(f"ADC sampling: {self._adc_buffer_size} samples @ 22050Hz")
        self._update_telemetry()
    
    def process_frame(self, predictions: Dict[str, float], inference_time_ms: float) -> InferenceResult:
        """
        Simulate TinyML inference on captured frame.
        
        Args:
            predictions: dict of class probabilities from ML model
            inference_time_ms: time taken for actual inference
            
        Returns:
            InferenceResult with alert level determination
        """
        self._set_state(NodeState.PROCESSING)
        self._log("Running TinyML inference...")
        self._log(f"Inference time: {inference_time_ms:.1f}ms")
        
        # Determine predicted class
        predicted_class = max(predictions, key=predictions.get)
        confidence = predictions[predicted_class]
        
        # Determine alert level
        alert_level = self._determine_alert(predicted_class, confidence)
        self.alert_level = alert_level
        
        result = InferenceResult(
            predictions=predictions,
            predicted_class=predicted_class,
            confidence=confidence,
            inference_time_ms=inference_time_ms,
            alert_level=alert_level,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            node_id=self.node_id
        )
        
        self.telemetry.total_inferences += 1
        self._inference_history.append(result)
        if len(self._inference_history) > 100:
            self._inference_history = self._inference_history[-80:]
        
        # Log results
        pred_str = " | ".join([f"{k}: {v:.3f}" for k, v in predictions.items()])
        self._log(f"Result: [{pred_str}]")
        self._log(f"Classification: {predicted_class.upper()} ({confidence:.1%}) → {alert_level.value}")
        
        if alert_level in (AlertLevel.INTRUSION, AlertLevel.HIGH_ALERT, AlertLevel.CRITICAL):
            self.telemetry.total_alerts += 1
            self._log(f"⚠ ALERT TRIGGERED: {alert_level.value}")
        
        self._update_telemetry()
        self._set_state(NodeState.IDLE)
        return result
    
    def _determine_alert(self, predicted_class: str, confidence: float) -> AlertLevel:
        """Apply decision engine threshold logic."""
        predicted_lower = predicted_class.lower()
        
        if "gunshot" in predicted_lower or "balastic" in predicted_lower:
            if confidence > 0.80:
                return AlertLevel.CRITICAL
            elif confidence > 0.70:
                return AlertLevel.HIGH_ALERT
            elif confidence > 0.50:
                return AlertLevel.SUSPICIOUS
        
        if "footstep" in predicted_lower:
            if confidence > 0.90:
                return AlertLevel.INTRUSION
            elif confidence > 0.85:
                return AlertLevel.INTRUSION
            elif confidence > 0.70:
                return AlertLevel.SUSPICIOUS
        
        return AlertLevel.NORMAL
    
    def prepare_transmission(self):
        """Set state to transmitting before LoRa TX."""
        self._set_state(NodeState.TRANSMITTING)
        self.telemetry.total_packets_sent += 1
        self._log("Preparing LoRa transmission...")
        self._update_telemetry()
    
    def enter_listening(self):
        """Enter low-power listening mode (wake-on-sound)."""
        self._set_state(NodeState.LISTENING)
        self._log("Entering wake-on-sound listening mode...")
        self._update_telemetry()
    
    def return_to_idle(self):
        """Return to idle state after processing cycle."""
        self._set_state(NodeState.IDLE)
        self._update_telemetry()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive node status."""
        self._update_telemetry()
        return {
            "node_id": self.node_id,
            "state": self.state.value,
            "alert_level": self.alert_level.value,
            "battery_percent": round(self.telemetry.battery_percent, 1),
            "temperature_c": round(self.telemetry.temperature_c, 1),
            "uptime_seconds": round(self.telemetry.uptime_seconds, 1),
            "total_inferences": self.telemetry.total_inferences,
            "total_alerts": self.telemetry.total_alerts,
            "total_packets": self.telemetry.total_packets_sent,
            "cpu_percent": round(self.telemetry.cpu_usage_percent, 1),
            "memory_percent": round(self.telemetry.memory_usage_percent, 1),
            "position": self.position,
            "is_running": self._is_running,
        }
    
    def get_logs(self, n: int = 50) -> list:
        """Get recent log entries."""
        return self._state_log[-n:]
    
    def get_inference_history(self, n: int = 20) -> list:
        """Get recent inference results."""
        return self._inference_history[-n:]
