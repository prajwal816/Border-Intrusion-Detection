"""
Decision Engine — Alert Classification & Threshold Logic
=========================================================
Applies security decision logic to classify inference results
into actionable alert levels for the border surveillance system.
"""

import time
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat classification levels."""
    CLEAR = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AlertEvent:
    """Structured alert event."""
    timestamp: str
    node_id: str
    event_type: str
    confidence: float
    threat_level: ThreatLevel
    description: str
    acknowledged: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "event_type": self.event_type,
            "confidence": round(self.confidence, 3),
            "threat_level": self.threat_level.name,
            "description": self.description,
            "acknowledged": self.acknowledged,
        }


class DecisionEngine:
    """
    Security decision engine applying threshold-based alert logic.
    
    Rules:
    - Footstep > 0.85 → INTRUSION DETECTED (HIGH)
    - Footstep > 0.70 → SUSPICIOUS ACTIVITY (MODERATE) 
    - Gunshot > 0.70  → HIGH ALERT (CRITICAL)
    - Gunshot > 0.50  → SUSPICIOUS (HIGH)
    - Noise (any)     → CLEAR
    
    Features:
    - Alert cooldown to prevent spamming
    - Consecutive detection escalation
    - Alert history management
    """
    
    # Thresholds
    FOOTSTEP_INTRUSION = 0.85
    FOOTSTEP_SUSPICIOUS = 0.70
    GUNSHOT_CRITICAL = 0.70
    GUNSHOT_SUSPICIOUS = 0.50
    
    # Cooldown
    ALERT_COOLDOWN_SECONDS = 3.0
    ESCALATION_WINDOW = 10.0      # seconds for consecutive detection
    ESCALATION_COUNT = 3           # consecutive alerts to escalate
    
    def __init__(self):
        self._alert_history: deque = deque(maxlen=500)
        self._last_alert_time: Dict[str, float] = {}
        self._consecutive_counts: Dict[str, int] = {}
        self._consecutive_timestamps: Dict[str, List[float]] = {}
        self._threat_level = ThreatLevel.CLEAR
        self._log_entries: deque = deque(maxlen=200)
    
    def evaluate(self, predictions: Dict[str, float], node_id: str = "01") -> AlertEvent:
        """
        Evaluate inference predictions and generate alert.
        
        Args:
            predictions: {class_name: probability} dict
            node_id: Node identifier
            
        Returns:
            AlertEvent with threat assessment
        """
        now = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Normalize prediction keys
        norm_preds = {}
        for k, v in predictions.items():
            key = k.lower().strip()
            norm_preds[key] = v
        
        # Find dominant class
        dominant_class = max(norm_preds, key=norm_preds.get)
        confidence = norm_preds[dominant_class]
        
        # Default
        threat_level = ThreatLevel.CLEAR
        description = "Environment normal — no threats detected"
        event_type = "noise"
        
        # Gunshot detection (highest priority)
        gunshot_conf = norm_preds.get("gunshot", norm_preds.get("balastic", 0.0))
        if gunshot_conf > self.GUNSHOT_CRITICAL:
            threat_level = ThreatLevel.CRITICAL
            event_type = "gunshot"
            confidence = gunshot_conf
            description = f"🚨 GUNSHOT DETECTED — Confidence: {gunshot_conf:.0%}"
        elif gunshot_conf > self.GUNSHOT_SUSPICIOUS:
            threat_level = ThreatLevel.HIGH
            event_type = "gunshot"
            confidence = gunshot_conf
            description = f"⚠ Possible gunshot — Confidence: {gunshot_conf:.0%}"
        
        # Footstep detection
        footstep_conf = norm_preds.get("footstep", norm_preds.get("footsteps", 0.0))
        if threat_level.value < ThreatLevel.HIGH.value:  # Don't downgrade from gunshot
            if footstep_conf > self.FOOTSTEP_INTRUSION:
                threat_level = ThreatLevel.HIGH
                event_type = "footstep"
                confidence = footstep_conf
                description = f"🚶 INTRUSION DETECTED — Footstep confidence: {footstep_conf:.0%}"
            elif footstep_conf > self.FOOTSTEP_SUSPICIOUS:
                threat_level = ThreatLevel.MODERATE
                event_type = "footstep"
                confidence = footstep_conf
                description = f"👁 Suspicious activity — Footstep confidence: {footstep_conf:.0%}"
        
        # Consecutive detection escalation
        if threat_level.value >= ThreatLevel.MODERATE.value:
            key = f"{node_id}_{event_type}"
            if key not in self._consecutive_timestamps:
                self._consecutive_timestamps[key] = []
            
            # Clean old timestamps
            self._consecutive_timestamps[key] = [
                t for t in self._consecutive_timestamps[key]
                if now - t < self.ESCALATION_WINDOW
            ]
            self._consecutive_timestamps[key].append(now)
            
            if len(self._consecutive_timestamps[key]) >= self.ESCALATION_COUNT:
                if threat_level.value < ThreatLevel.CRITICAL.value:
                    threat_level = ThreatLevel(min(threat_level.value + 1, ThreatLevel.CRITICAL.value))
                    description += f" [ESCALATED — {len(self._consecutive_timestamps[key])} consecutive detections]"
        
        # Cooldown check
        cooldown_key = f"{node_id}_{event_type}"
        last_time = self._last_alert_time.get(cooldown_key, 0)
        is_cooldown = (now - last_time) < self.ALERT_COOLDOWN_SECONDS and threat_level.value >= ThreatLevel.MODERATE.value
        
        if not is_cooldown and threat_level.value >= ThreatLevel.MODERATE.value:
            self._last_alert_time[cooldown_key] = now
        
        # Create event
        alert = AlertEvent(
            timestamp=timestamp,
            node_id=node_id,
            event_type=event_type,
            confidence=confidence,
            threat_level=threat_level,
            description=description,
        )
        
        self._alert_history.append(alert)
        self._threat_level = threat_level
        
        # Log
        log_msg = f"[DECISION] Node-{node_id}: {event_type.upper()} ({confidence:.1%}) → {threat_level.name}"
        self._log_entries.append(f"[{timestamp}] {log_msg}")
        logger.info(log_msg)
        
        return alert
    
    def get_current_threat_level(self) -> ThreatLevel:
        return self._threat_level
    
    def get_alert_history(self, n: int = 30) -> List[Dict]:
        return [a.to_dict() for a in list(self._alert_history)[-n:]]
    
    def get_active_alerts(self, n: int = 10) -> List[Dict]:
        """Get recent high-priority alerts."""
        high_alerts = [
            a for a in self._alert_history
            if a.threat_level.value >= ThreatLevel.MODERATE.value
        ]
        return [a.to_dict() for a in list(high_alerts)[-n:]]
    
    def get_logs(self, n: int = 20) -> List[str]:
        return list(self._log_entries)[-n:]
