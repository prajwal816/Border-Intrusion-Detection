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
import random
import numpy as np
import streamlit as st
from datetime import datetime

# Add project root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

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

MODEL_ONNX = os.path.join(_SCRIPT_DIR, "models", "border_intrusion_model.onnx")
DATASET_DIR = os.path.join(_SCRIPT_DIR, "dataset")


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "system_running": False,
        "audio_mode": "live",
        "current_frame": np.zeros(22050, dtype=np.float32),
        "current_predictions": {"footstep": 0.0, "gunshot": 0.0, "noise": 1.0},
        "smoothed_predictions": {"footstep": 0.0, "gunshot": 0.0, "noise": 1.0},
        "prediction_history": [],
        "nodes_initialized": False,
        "node": None,
        "transmitter": None,
        "base_station": None,
        "decision_engine": None,
        "classifier": None,
        "audio_capture": None,
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
    
    # Initialize single node
    node = VirtualESP32Node(node_id="01", position=(5, 5))
    node.boot()
    st.session_state.node = node
    st.session_state.transmitter = LoRaTransmitter(node_id="01")
    
    # Base station
    st.session_state.base_station = BaseStationReceiver(station_id="BASE-01")
    
    # Decision engine
    st.session_state.decision_engine = DecisionEngine()
    
    # ML Model — ONNX Runtime (no TensorFlow needed)
    with st.spinner("Loading TinyML model..."):
        if os.path.exists(MODEL_ONNX):
            print(f"[INIT] Loading ONNX model: {MODEL_ONNX}")
            st.session_state.classifier = AudioClassifier(model_path=MODEL_ONNX)
        else:
            print(f"[INIT] ❌ No ONNX model found at {MODEL_ONNX}")
            st.session_state.classifier = AudioClassifier(model_path=None)
    
    print(f"[INIT] Final model type: {st.session_state.classifier.model_type}")
    
    # Audio capture — LIVE from laptop microphone
    audio_mode = "live" if SOUNDDEVICE_AVAILABLE else "replay"
    st.session_state.audio_mode = audio_mode
    st.session_state.audio_capture = AudioCapture(
        mode=audio_mode,
        replay_dir=DATASET_DIR
    )
    
    st.session_state.nodes_initialized = True
    st.session_state.system_start_time = time.time()
    logger.info(f"System initialized — Model: {st.session_state.classifier.model_type}, Audio: {audio_mode}")


def run_detection_cycle():
    """Run one complete detection cycle: capture → inference → decision → transmit."""
    audio_capture = st.session_state.audio_capture
    classifier = st.session_state.classifier
    node = st.session_state.node
    transmitter = st.session_state.transmitter
    base_station = st.session_state.base_station
    decision_engine = st.session_state.decision_engine
    
    # 1. Capture audio frame
    node.enter_listening()
    frame = audio_capture.get_frame(timeout=1.5)
    
    if frame is None:
        frame = np.random.randn(22050).astype(np.float32) * 0.001
    
    st.session_state.current_frame = frame
    
    # 2. Compute signal metrics
    rms = float(np.sqrt(np.mean(frame**2)))
    peak = float(np.max(np.abs(frame)))
    
    # 3. Capture & process
    node.capture_audio_frame()
    
    # 4. Extract features & run inference
    features = audio_capture.preprocess_for_model(frame)
    raw_predictions = classifier.predict(features)
    
    # 5. Energy-based confidence weighting
    # Quiet signal → bias toward noise (model can't distinguish ambient noise)
    # Loud signal → trust the model fully
    QUIET_THRESHOLD = 0.005   # Below this, force noise
    LOUD_THRESHOLD = 0.05     # Above this, trust model fully
    
    noise_prior = {"footstep": 0.02, "gunshot": 0.01, "noise": 0.97}
    
    if rms < QUIET_THRESHOLD:
        # Very quiet — definitely noise
        model_weight = 0.0
    elif rms > LOUD_THRESHOLD:
        # Loud — trust the model
        model_weight = 1.0
    else:
        # Blend: scale linearly from 0 to 1
        model_weight = (rms - QUIET_THRESHOLD) / (LOUD_THRESHOLD - QUIET_THRESHOLD)
    
    predictions = {}
    for k in raw_predictions:
        predictions[k] = model_weight * raw_predictions[k] + (1 - model_weight) * noise_prior.get(k, 0.0)
    
    # Debug output
    dominant = max(predictions, key=predictions.get)
    raw_dom = max(raw_predictions, key=raw_predictions.get)
    print(f"[CYCLE {st.session_state.total_cycles:03d}] RMS={rms:.6f} Peak={peak:.4f} weight={model_weight:.2f} raw={raw_dom} → {dominant}({predictions[dominant]:.1%})")
    
    inference_time = classifier.get_last_inference_time()
    
    # 6. Smooth predictions (exponential moving average)
    SMOOTH_ALPHA = 0.3  # Lower = smoother
    prev = st.session_state.smoothed_predictions
    smoothed = {}
    for k in predictions:
        smoothed[k] = SMOOTH_ALPHA * predictions[k] + (1 - SMOOTH_ALPHA) * prev.get(k, 0.0)
    st.session_state.smoothed_predictions = smoothed
    st.session_state.current_predictions = smoothed
    
    st.session_state.prediction_history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "predictions": smoothed.copy(),
    })
    if len(st.session_state.prediction_history) > 50:
        st.session_state.prediction_history = st.session_state.prediction_history[-30:]
    
    # 6. Node processes inference
    result = node.process_frame(smoothed, inference_time)
    
    # 7. Decision engine
    alert = decision_engine.evaluate(smoothed, node_id="01")
    
    # 8. LoRa transmission
    node.prepare_transmission()
    packet = transmitter.transmit(result, node.get_status())
    
    # 9. Base station receives
    base_alert = base_station.receive_packet(packet)
    
    # Store signal metrics
    st.session_state.last_rssi = packet.rssi_dbm
    st.session_state.last_snr = packet.snr_db
    st.session_state.last_latency = packet.latency_ms
    
    node.return_to_idle()
    st.session_state.total_cycles += 1


# ──────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────

def main():
    init_session_state()
    initialize_system()
    inject_custom_css()
    
    # Get classifier from session state
    classifier = st.session_state.classifier
    
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
    
    # Row 1: Node Status Card (single node)
    st.markdown("## ◈ Node Status")
    node_placeholder = st.empty()
    
    st.markdown("---")
    
    # Row 2: Live Audio + Classification (updates smoothly via fragment)
    col_wave, col_class = st.columns([2, 1])
    with col_wave:
        st.markdown("## ◈ Live Audio Waveform")
        waveform_placeholder = st.empty()
    with col_class:
        st.markdown("## ◈ Classification")
        class_placeholder = st.empty()
        result_placeholder = st.empty()
    
    st.markdown("---")
    
    # Row 3: Alerts + Deployment Map
    col_alerts, col_map = st.columns([1, 1])
    with col_alerts:
        st.markdown("## ◈ Alert Panel")
        alert_placeholder = st.empty()
    with col_map:
        st.markdown("## ◈ Deployment Map")
        map_placeholder = st.empty()
    
    st.markdown("---")
    
    # Row 4: Communication Logs
    st.markdown("## ◈ Communication Log")
    col_tx, col_rx = st.columns(2)
    with col_tx:
        st.markdown("### LoRa TX Log")
        tx_placeholder = st.empty()
    with col_rx:
        st.markdown("### Base Station RX Log")
        rx_placeholder = st.empty()
    
    # ── Live Update Loop ──
    # Use st.fragment for smooth in-place updates (no full-page blink)
    @st.fragment(run_every=1)
    def live_update():
        """Fragment that updates dynamic content in-place every 1 second."""
        if st.session_state.system_running:
            run_detection_cycle()
        
        # Update node card
        node = st.session_state.node
        if node:
            with node_placeholder.container():
                render_node_card(node.get_status())
        
        # Update waveform (in-place, no blink)
        with waveform_placeholder.container():
            waveform_fig = render_waveform(st.session_state.current_frame)
            st.plotly_chart(waveform_fig, use_container_width=True, key=f"wf_{st.session_state.total_cycles}")
        
        # Update classification bars
        with class_placeholder.container():
            class_fig = render_classification_bars(st.session_state.current_predictions)
            st.plotly_chart(class_fig, use_container_width=True, key=f"cl_{st.session_state.total_cycles}")
        
        # Update detection result badge
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
            
            with result_placeholder.container():
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: #131920; border: 1px solid rgba(57,255,20,0.25);
                            border-radius: 8px; margin-top: 10px;">
                    <div style="font-family: 'Orbitron', monospace; font-size: 1.2rem; color: {color};">
                        {icon} {dominant.upper()}
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #6b7280;">
                        Confidence: {conf:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Update alerts
        decision_engine = st.session_state.decision_engine
        if decision_engine:
            with alert_placeholder.container():
                alerts = decision_engine.get_active_alerts(15)
                render_alert_panel(alerts)
        
        # Update map
        node = st.session_state.node
        with map_placeholder.container():
            node_statuses = [node.get_status()] if node else []
            map_fig = render_deployment_map(node_statuses)
            st.plotly_chart(map_fig, use_container_width=True, key=f"mp_{st.session_state.total_cycles}")
        
        # Update comm logs
        tx = st.session_state.transmitter
        with tx_placeholder.container():
            render_comm_log(tx.get_logs(20) if tx else [])
        
        base_station = st.session_state.base_station
        with rx_placeholder.container():
            if base_station:
                render_comm_log(base_station.get_logs(20))
    
    # Run the fragment
    live_update()


if __name__ == "__main__":
    main()
