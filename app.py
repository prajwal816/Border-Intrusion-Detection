"""
Border Sentinel — Edge AI Border Intrusion Detection Dashboard
================================================================
Production surveillance dashboard simulating ESP32 + MEMS + LoRa
border security network with real-time acoustic classification.

Run: streamlit run app.py
"""

import os
import sys
import time
import threading
import logging
import numpy as np
import streamlit as st
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.audio import AudioCapture, SOUNDDEVICE_AVAILABLE
from src.model import AudioClassifier
from src.node import VirtualESP32Node, AlertLevel
from src.communication import LoRaTransmitter, BaseStationReceiver
from src.decision import DecisionEngine
from src.ui_components import (
    inject_custom_css, render_header, render_node_card,
    render_waveform, render_classification_bars,
    render_alert_panel, render_comm_log,
    render_deployment_map, render_signal_gauge, THEME
)

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="BORDER SENTINEL | Edge AI Surveillance",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Model & System Paths
# ──────────────────────────────────────────────

# Resolve paths robustly for Streamlit
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_H5 = os.path.join(_SCRIPT_DIR, "models", "border_intrusion_model.h5")
MODEL_TFLITE = os.path.join(_SCRIPT_DIR, "models", "border_intrusion_model.tflite")
DATASET_DIR = os.path.join(_SCRIPT_DIR, "dataset")

logger.info(f"[PATHS] Script dir: {_SCRIPT_DIR}")
logger.info(f"[PATHS] Model H5 exists: {os.path.exists(MODEL_H5)} → {MODEL_H5}")
logger.info(f"[PATHS] Model TFLite exists: {os.path.exists(MODEL_TFLITE)} → {MODEL_TFLITE}")
logger.info(f"[PATHS] Dataset exists: {os.path.isdir(DATASET_DIR)} → {DATASET_DIR}")


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "system_running": False,
        "audio_mode": "replay",
        "current_frame": np.zeros(22050, dtype=np.float32),
        "current_predictions": {"footstep": 0.0, "gunshot": 0.0, "noise": 0.0},
        "prediction_history": [],
        "nodes_initialized": False,
        "nodes": {},
        "transmitters": {},
        "base_station": None,
        "decision_engine": None,
        "classifier": None,
        "audio_capture": None,
        "all_comm_logs": [],
        "system_start_time": None,
        "total_cycles": 0,
        "last_rssi": -70.0,
        "last_snr": 10.0,
        "last_latency": 100.0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def initialize_system():
    """Boot all system components."""
    if st.session_state.nodes_initialized:
        return
    
    # Initialize nodes
    positions = [(2, 8), (8, 7), (5, 2)]
    for i in range(3):
        node_id = f"{i+1:02d}"
        node = VirtualESP32Node(node_id=node_id, position=positions[i])
        node.boot()
        st.session_state.nodes[node_id] = node
        st.session_state.transmitters[node_id] = LoRaTransmitter(node_id=node_id)
    
    # Base station
    st.session_state.base_station = BaseStationReceiver(station_id="BASE-01")
    
    # Decision engine
    st.session_state.decision_engine = DecisionEngine()
    
    # ML Model — prefer H5 for reliability, fall back to TFLite
    if os.path.exists(MODEL_H5):
        model_path = MODEL_H5
    elif os.path.exists(MODEL_TFLITE):
        model_path = MODEL_TFLITE
    else:
        model_path = None
    logger.info(f"[INIT] Loading model from: {model_path}")
    st.session_state.classifier = AudioClassifier(model_path=model_path)
    
    # Audio capture
    audio_mode = "live" if SOUNDDEVICE_AVAILABLE else "replay"
    st.session_state.audio_mode = audio_mode
    st.session_state.audio_capture = AudioCapture(
        mode=audio_mode,
        replay_dir=DATASET_DIR
    )
    
    st.session_state.nodes_initialized = True
    st.session_state.system_start_time = time.time()
    logger.info("System initialized successfully")


def run_detection_cycle():
    """Run one complete detection cycle: capture → inference → decision → transmit."""
    audio_capture = st.session_state.audio_capture
    classifier = st.session_state.classifier
    node = st.session_state.nodes["01"]  # Primary node uses live audio
    transmitter = st.session_state.transmitters["01"]
    base_station = st.session_state.base_station
    decision_engine = st.session_state.decision_engine
    
    # 1. Capture audio frame
    node.enter_listening()
    frame = audio_capture.get_frame(timeout=1.5)
    
    if frame is None:
        frame = np.random.randn(22050).astype(np.float32) * 0.01
    
    st.session_state.current_frame = frame
    
    # 2. Check wake-on-sound
    energy = audio_capture.compute_energy(frame)
    
    # 3. Capture & process
    node.capture_audio_frame()
    
    # 4. Extract features & run inference
    features = audio_capture.preprocess_for_model(frame)
    predictions = classifier.predict(features)
    inference_time = classifier.get_last_inference_time()
    
    st.session_state.current_predictions = predictions
    st.session_state.prediction_history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "predictions": predictions.copy(),
    })
    if len(st.session_state.prediction_history) > 50:
        st.session_state.prediction_history = st.session_state.prediction_history[-30:]
    
    # 5. Node processes inference
    result = node.process_frame(predictions, inference_time)
    
    # 6. Decision engine
    alert = decision_engine.evaluate(predictions, node_id="01")
    
    # 7. LoRa transmission
    node.prepare_transmission()
    packet = transmitter.transmit(result, node.get_status())
    
    # 8. Base station receives
    base_alert = base_station.receive_packet(packet)
    
    # Store signal metrics
    st.session_state.last_rssi = packet.rssi_dbm
    st.session_state.last_snr = packet.snr_db
    st.session_state.last_latency = packet.latency_ms
    
    # Collect comm logs
    tx_logs = transmitter.get_logs(5)
    rx_logs = base_station.get_logs(5)
    st.session_state.all_comm_logs = tx_logs + rx_logs
    
    node.return_to_idle()
    st.session_state.total_cycles += 1
    
    # --- Simulate other nodes with random dataset samples ---
    _simulate_secondary_nodes()


def _simulate_secondary_nodes():
    """Simulate activity for Node 02 and Node 03."""
    import random
    
    for node_id in ["02", "03"]:
        node = st.session_state.nodes[node_id]
        transmitter = st.session_state.transmitters[node_id]
        base_station = st.session_state.base_station
        decision_engine = st.session_state.decision_engine
        
        # Random activity (50% chance each cycle)
        if random.random() > 0.5:
            node.enter_listening()
            node.capture_audio_frame()
            
            # Generate random predictions (mostly noise for secondary nodes)
            r = random.random()
            if r > 0.92:
                # Rare footstep event
                preds = {"footstep": random.uniform(0.6, 0.95),
                         "gunshot": random.uniform(0.01, 0.1),
                         "noise": random.uniform(0.05, 0.2)}
            elif r > 0.97:
                # Very rare gunshot event
                preds = {"footstep": random.uniform(0.01, 0.1),
                         "gunshot": random.uniform(0.7, 0.98),
                         "noise": random.uniform(0.01, 0.1)}
            else:
                # Normal noise
                preds = {"footstep": random.uniform(0.01, 0.15),
                         "gunshot": random.uniform(0.01, 0.08),
                         "noise": random.uniform(0.75, 0.95)}
            
            # Normalize
            total = sum(preds.values())
            preds = {k: v/total for k, v in preds.items()}
            
            inference_time = random.uniform(15, 45)
            result = node.process_frame(preds, inference_time)
            
            decision_engine.evaluate(preds, node_id=node_id)
            
            node.prepare_transmission()
            packet = transmitter.transmit(result, node.get_status())
            base_station.receive_packet(packet)
            
            node.return_to_idle()


# ──────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────

def main():
    init_session_state()
    initialize_system()
    inject_custom_css()
    
    # ── Header ──
    render_header()
    
    # ── Sidebar ──
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <div style="font-family: 'Orbitron', monospace; font-size: 1rem; color: #39FF14;
                        text-shadow: 0 0 15px rgba(57,255,20,0.4);">
                ⬡ CONTROL PANEL
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # System control
        st.markdown("### System Control")
        
        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("▶ START", use_container_width=True, type="primary"):
                st.session_state.system_running = True
                if st.session_state.audio_capture and not st.session_state.audio_capture.is_running:
                    st.session_state.audio_capture.start()
        with col_stop:
            if st.button("■ STOP", use_container_width=True):
                st.session_state.system_running = False
                if st.session_state.audio_capture:
                    st.session_state.audio_capture.stop()
        
        # System info
        st.markdown("---")
        st.markdown("### System Info")
        
        classifier = st.session_state.classifier
        if classifier:
            model_info = classifier.get_model_info()
            st.markdown(f"""
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #6b7280;">
                <div>Model: <span style="color: #39FF14;">{model_info['model_type'].upper()}</span></div>
                <div>Classes: <span style="color: #c5c8c6;">{', '.join(model_info['classes'])}</span></div>
                <div>Avg Inference: <span style="color: #FFB000;">{model_info['avg_inference_ms']:.1f}ms</span></div>
                <div>Audio Mode: <span style="color: #00D4FF;">{st.session_state.audio_mode.upper()}</span></div>
            </div>
            """, unsafe_allow_html=True)
        
        # Stats
        st.markdown("---")
        st.markdown("### Statistics")
        
        base_station = st.session_state.base_station
        if base_station:
            stats = base_station.get_stats()
            uptime = time.time() - st.session_state.system_start_time if st.session_state.system_start_time else 0
            mins, secs = divmod(int(uptime), 60)
            hours, mins = divmod(mins, 60)
            
            st.metric("Uptime", f"{hours:02d}:{mins:02d}:{secs:02d}")
            st.metric("Total Packets", stats["total_packets"])
            st.metric("Detection Cycles", st.session_state.total_cycles)
            st.metric("Active Alerts", stats["total_alerts"])
            st.metric("Delivery Rate", f"{stats['delivery_rate']}%")
        
        # Signal strength
        st.markdown("---")
        st.markdown("### Signal Quality")
        render_signal_gauge(st.session_state.last_rssi, st.session_state.last_snr)
        st.markdown(f"""
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #6b7280; margin-top: 8px;">
            Latency: <span style="color: #FFB000;">{st.session_state.last_latency:.0f}ms</span>
        </div>
        """, unsafe_allow_html=True)
    
    # ── Main Content ──
    
    # Run detection if system is active
    if st.session_state.system_running:
        run_detection_cycle()
    
    # Row 1: Node Status Cards
    st.markdown("## ◈ Node Network Status")
    node_cols = st.columns(3)
    for i, (node_id, node) in enumerate(st.session_state.nodes.items()):
        with node_cols[i]:
            render_node_card(node.get_status())
    
    st.markdown("---")
    
    # Row 2: Live Audio + Classification
    col_wave, col_class = st.columns([2, 1])
    
    with col_wave:
        st.markdown("## ◈ Live Audio Waveform")
        waveform_fig = render_waveform(st.session_state.current_frame)
        st.plotly_chart(waveform_fig, use_container_width=True, key="waveform")
    
    with col_class:
        st.markdown("## ◈ Classification")
        class_fig = render_classification_bars(st.session_state.current_predictions)
        st.plotly_chart(class_fig, use_container_width=True, key="classification")
        
        # Current detection result
        preds = st.session_state.current_predictions
        if preds:
            dominant = max(preds, key=preds.get)
            conf = preds[dominant]
            
            if dominant.lower() in ("gunshot", "balastic") and conf > 0.5:
                color = "#FF3131"
                icon = "🔴"
            elif dominant.lower() in ("footstep", "footsteps") and conf > 0.7:
                color = "#FFB000"
                icon = "🟡"
            else:
                color = "#39FF14"
                icon = "🟢"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; background: #131920; border: 1px solid {color}44;
                        border-radius: 8px; margin-top: 10px;">
                <div style="font-family: 'Orbitron', monospace; font-size: 1.2rem; color: {color};
                            text-shadow: 0 0 15px {color}66;">
                    {icon} {dominant.upper()}
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #6b7280;">
                    Confidence: {conf:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Row 3: Alerts + Deployment Map
    col_alerts, col_map = st.columns([1, 1])
    
    with col_alerts:
        st.markdown("## ◈ Alert Panel")
        decision_engine = st.session_state.decision_engine
        if decision_engine:
            alerts = decision_engine.get_active_alerts(15)
            render_alert_panel(alerts)
    
    with col_map:
        st.markdown("## ◈ Deployment Map")
        node_statuses = [node.get_status() for node in st.session_state.nodes.values()]
        map_fig = render_deployment_map(node_statuses)
        st.plotly_chart(map_fig, use_container_width=True, key="map")
    
    st.markdown("---")
    
    # Row 4: Communication Logs
    st.markdown("## ◈ Communication Log")
    col_tx, col_rx = st.columns(2)
    
    with col_tx:
        st.markdown("### LoRa TX Log")
        all_tx = []
        for tx in st.session_state.transmitters.values():
            all_tx.extend(tx.get_logs(10))
        all_tx.sort()
        render_comm_log(all_tx[-20:])
    
    with col_rx:
        st.markdown("### Base Station RX Log")
        base_station = st.session_state.base_station
        if base_station:
            render_comm_log(base_station.get_logs(20))
    
    # Auto-refresh when running
    if st.session_state.system_running:
        time.sleep(0.3)
        st.rerun()


if __name__ == "__main__":
    main()
