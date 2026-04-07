"""
LoRa Communication Simulation — Wireless Transmission Layer
============================================================
Simulates LoRa SX1276 radio communication between edge nodes
and a central base station for border surveillance.
"""

import time
import random
import threading
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class LoRaPacket:
    """Simulated LoRa radio packet."""
    packet_id: int = 0
    node_id: str = ""
    event_type: str = ""
    confidence: float = 0.0
    alert_level: str = ""
    predictions: Dict[str, float] = field(default_factory=dict)
    battery_percent: float = 0.0
    temperature_c: float = 0.0
    rssi_dbm: float = -70.0
    snr_db: float = 10.0
    spreading_factor: int = 7
    bandwidth_khz: int = 125
    tx_power_dbm: int = 14
    timestamp: str = ""
    latency_ms: float = 0.0
    crc_valid: bool = True
    payload_size_bytes: int = 0
    
    def format_payload(self) -> str:
        """Format packet payload as string."""
        return (
            f"NODE_ID={self.node_id} | "
            f"EVENT={self.event_type.upper()} | "
            f"CONF={self.confidence:.2f} | "
            f"ALERT={self.alert_level} | "
            f"BAT={self.battery_percent:.0f}% | "
            f"RSSI={self.rssi_dbm:.0f}dBm"
        )


class LoRaTransmitter:
    """
    Simulates LoRa SX1276 radio transmitter on an ESP32 node.
    
    Features:
    - Packet formation with headers, payload, and CRC
    - Configurable spreading factor, bandwidth, TX power
    - Simulated RSSI, SNR, and packet latency
    - Channel activity detection (Listen-Before-Talk)
    """
    
    def __init__(self, node_id: str = "01"):
        self.node_id = node_id
        self._packet_counter = 0
        self._tx_log = deque(maxlen=200)
        self._lock = threading.Lock()
        
        # Radio parameters
        self.spreading_factor = 7
        self.bandwidth_khz = 125
        self.tx_power_dbm = 14
        self.frequency_mhz = 915.0  # US ISM band
        
        self._log(f"LoRa TX initialized — SF{self.spreading_factor}, BW{self.bandwidth_khz}kHz, {self.tx_power_dbm}dBm")
    
    def _log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{timestamp}] [LoRa TX NODE-{self.node_id}] {message}"
        with self._lock:
            self._tx_log.append(entry)
        logger.info(entry)
    
    def transmit(self, inference_result, node_telemetry: Dict = None) -> LoRaPacket:
        """
        Simulate LoRa packet transmission.
        
        Args:
            inference_result: InferenceResult from node processing
            node_telemetry: Optional telemetry data
            
        Returns:
            LoRaPacket with simulated radio metrics
        """
        self._packet_counter += 1
        
        # Simulate radio characteristics
        base_rssi = -60.0
        distance_factor = random.uniform(0, 40)
        rssi = base_rssi - distance_factor + random.gauss(0, 3)
        snr = 12.0 - (distance_factor / 5) + random.gauss(0, 1.5)
        
        # Calculate airtime / latency
        payload_size = 32 + len(str(inference_result.predictions)) 
        symbol_duration = (2 ** self.spreading_factor) / (self.bandwidth_khz * 1000)
        base_latency = symbol_duration * payload_size * 8 * 1000  # ms
        latency = max(50, base_latency + random.uniform(20, 150))
        
        # CRC simulation (99.5% success rate)
        crc_valid = random.random() > 0.005
        
        packet = LoRaPacket(
            packet_id=self._packet_counter,
            node_id=self.node_id,
            event_type=inference_result.predicted_class,
            confidence=inference_result.confidence,
            alert_level=inference_result.alert_level.value,
            predictions=inference_result.predictions.copy(),
            battery_percent=node_telemetry.get("battery_percent", 100) if node_telemetry else 100,
            temperature_c=node_telemetry.get("temperature_c", 25) if node_telemetry else 25,
            rssi_dbm=round(rssi, 1),
            snr_db=round(snr, 1),
            spreading_factor=self.spreading_factor,
            bandwidth_khz=self.bandwidth_khz,
            tx_power_dbm=self.tx_power_dbm,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            latency_ms=round(latency, 1),
            crc_valid=crc_valid,
            payload_size_bytes=payload_size,
        )
        
        self._log(f"Sending packet #{self._packet_counter}...")
        self._log(f"Payload: {packet.format_payload()}")
        self._log(f"Radio: RSSI={rssi:.0f}dBm | SNR={snr:.1f}dB | Latency={latency:.0f}ms | CRC={'✓' if crc_valid else '✗'}")
        
        return packet
    
    def get_logs(self, n: int = 30) -> list:
        with self._lock:
            return list(self._tx_log)[-n:]


class BaseStationReceiver:
    """
    Simulates a LoRa base station / gateway that receives
    packets from multiple edge nodes.
    
    Features:
    - Multi-node packet reception
    - Alert aggregation & prioritization
    - Node registry & health monitoring
    - Packet delivery statistics
    """
    
    def __init__(self, station_id: str = "BASE-01"):
        self.station_id = station_id
        self._node_registry: Dict[str, Dict] = {}
        self._received_packets: deque = deque(maxlen=500)
        self._alerts: deque = deque(maxlen=200)
        self._rx_log: deque = deque(maxlen=300)
        self._stats = {
            "total_packets": 0,
            "total_alerts": 0,
            "packets_dropped": 0,
            "uptime_start": time.time(),
        }
        self._lock = threading.Lock()
        
        self._log(f"Base Station {self.station_id} online")
        self._log("Listening on 915.0 MHz, SF7-12 adaptive")
    
    def _log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{timestamp}] [BASE STATION] {message}"
        with self._lock:
            self._rx_log.append(entry)
        logger.info(entry)
    
    def receive_packet(self, packet: LoRaPacket) -> Optional[Dict]:
        """
        Process an incoming LoRa packet.
        
        Returns:
            Alert dict if the packet triggers an alert, else None
        """
        self._stats["total_packets"] += 1
        
        # Simulate packet drop on CRC failure
        if not packet.crc_valid:
            self._stats["packets_dropped"] += 1
            self._log(f"⚠ Packet #{packet.packet_id} from NODE-{packet.node_id}: CRC FAILED — dropped")
            return None
        
        self._log(f"Packet #{packet.packet_id} received from NODE-{packet.node_id}")
        self._log(f"  Event: {packet.event_type.upper()} | Confidence: {packet.confidence:.1%}")
        self._log(f"  RSSI: {packet.rssi_dbm}dBm | Latency: {packet.latency_ms}ms")
        
        with self._lock:
            self._received_packets.append(packet)
        
        # Update node registry
        self._node_registry[packet.node_id] = {
            "last_seen": datetime.now().strftime("%H:%M:%S"),
            "battery": packet.battery_percent,
            "rssi": packet.rssi_dbm,
            "temperature": packet.temperature_c,
            "last_event": packet.event_type,
            "packets_received": self._node_registry.get(packet.node_id, {}).get("packets_received", 0) + 1,
        }
        
        # Generate alert if needed
        alert = None
        if packet.alert_level in ("INTRUSION", "HIGH_ALERT", "CRITICAL"):
            alert = {
                "timestamp": packet.timestamp,
                "node_id": packet.node_id,
                "event": packet.event_type,
                "confidence": packet.confidence,
                "alert_level": packet.alert_level,
                "rssi": packet.rssi_dbm,
            }
            with self._lock:
                self._alerts.append(alert)
            self._stats["total_alerts"] += 1
            
            self._log("━" * 50)
            self._log(f"🚨 ALERT RECEIVED:")
            self._log(f"   Node: {packet.node_id}")
            self._log(f"   Event: {packet.event_type.upper()}")
            self._log(f"   Confidence: {packet.confidence:.0%}")
            self._log(f"   Level: {packet.alert_level}")
            self._log("━" * 50)
        
        return alert
    
    def get_alerts(self, n: int = 20) -> list:
        with self._lock:
            return list(self._alerts)[-n:]
    
    def get_logs(self, n: int = 30) -> list:
        with self._lock:
            return list(self._rx_log)[-n:]
    
    def get_node_registry(self) -> Dict:
        return self._node_registry.copy()
    
    def get_stats(self) -> Dict:
        stats = self._stats.copy()
        stats["uptime_seconds"] = round(time.time() - stats["uptime_start"], 1)
        stats["delivery_rate"] = (
            round((stats["total_packets"] - stats["packets_dropped"]) / max(1, stats["total_packets"]) * 100, 1)
        )
        return stats
    
    def get_recent_packets(self, n: int = 10) -> list:
        with self._lock:
            return list(self._received_packets)[-n:]
