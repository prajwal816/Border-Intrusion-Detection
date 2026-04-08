"""
UI Components — Military Surveillance Dashboard Widgets
========================================================
Reusable Streamlit components for the border intrusion
detection dashboard with military/surveillance aesthetic.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional


# ──────────────────────────────────────────────
# Theme Constants
# ──────────────────────────────────────────────

THEME = {
    "bg_primary": "#0a0e14",
    "bg_secondary": "#131920",
    "bg_card": "#1a2332",
    "accent_green": "#39FF14",
    "accent_amber": "#FFB000",
    "accent_red": "#FF3131",
    "accent_cyan": "#00D4FF",
    "accent_purple": "#A855F7",
    "text_primary": "#c5c8c6",
    "text_secondary": "#6b7280",
    "border": "#2a3a4a",
    "grid": "#1a2332",
}

ALERT_COLORS = {
    "CLEAR": THEME["accent_green"],
    "NORMAL": THEME["accent_green"],
    "LOW": THEME["accent_green"],
    "MODERATE": THEME["accent_amber"],
    "SUSPICIOUS": THEME["accent_amber"],
    "HIGH": THEME["accent_red"],
    "INTRUSION": THEME["accent_red"],
    "HIGH_ALERT": THEME["accent_red"],
    "CRITICAL": "#FF0040",
}


# ──────────────────────────────────────────────
# CSS Injection
# ──────────────────────────────────────────────

def inject_custom_css():
    """Inject military/surveillance theme CSS."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Orbitron:wght@400;500;600;700;800;900&display=swap');
    
    /* Global */
    .stApp {
        background-color: #0a0e14;
        background-image: 
            radial-gradient(circle at 25% 25%, rgba(57, 255, 20, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 75% 75%, rgba(255, 49, 49, 0.02) 0%, transparent 50%);
    }
    
    /* Scanline overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(57, 255, 20, 0.01) 2px,
            rgba(57, 255, 20, 0.01) 4px
        );
        pointer-events: none;
        z-index: 1000;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Orbitron', 'JetBrains Mono', monospace !important;
        color: #39FF14 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    h1 { 
        text-shadow: 0 0 20px rgba(57, 255, 20, 0.5), 0 0 40px rgba(57, 255, 20, 0.2);
        font-size: 1.8rem !important;
    }
    h2 { font-size: 1.1rem !important; }
    h3 { font-size: 0.95rem !important; }
    
    /* Text */
    p, span, label, .stMarkdown {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #131920 0%, #1a2332 100%);
        border: 1px solid #2a3a4a;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 0 15px rgba(57, 255, 20, 0.05), inset 0 1px 0 rgba(255,255,255,0.03);
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Orbitron', monospace !important;
        font-size: 0.7rem !important;
        color: #6b7280 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', monospace !important;
        color: #39FF14 !important;
        font-size: 1.5rem !important;
        text-shadow: 0 0 10px rgba(57, 255, 20, 0.3);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e14 0%, #0d1318 100%);
        border-right: 1px solid #39FF1433;
    }
    
    /* Cards / containers */
    .node-card {
        background: linear-gradient(135deg, #131920 0%, #1a2332 100%);
        border: 1px solid #2a3a4a;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .node-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, #39FF14, transparent);
    }
    
    .node-card.alert-red::before {
        background: linear-gradient(90deg, transparent, #FF3131, transparent);
        box-shadow: 0 0 20px rgba(255, 49, 49, 0.3);
    }
    
    .node-card.alert-amber::before {
        background: linear-gradient(90deg, transparent, #FFB000, transparent);
    }
    
    .alert-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-family: 'Orbitron', monospace;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .alert-green { background: rgba(57, 255, 20, 0.15); color: #39FF14; border: 1px solid #39FF1444; }
    .alert-amber { background: rgba(255, 176, 0, 0.15); color: #FFB000; border: 1px solid #FFB00044; }
    .alert-red { background: rgba(255, 49, 49, 0.15); color: #FF3131; border: 1px solid #FF313144; animation: pulse-red 2s infinite; }
    
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 5px rgba(255, 49, 49, 0.3); }
        50% { box-shadow: 0 0 20px rgba(255, 49, 49, 0.6); }
    }
    
    .comm-log {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #39FF14;
        background: #0a0e14;
        border: 1px solid #1a2332;
        border-radius: 6px;
        padding: 12px;
        max-height: 400px;
        overflow-y: auto;
        line-height: 1.6;
    }
    
    .map-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        padding: 20px;
        background: #0d1318;
        border: 1px solid #2a3a4a;
        border-radius: 10px;
    }
    
    .header-bar {
        background: linear-gradient(90deg, #0a0e14, #131920, #0a0e14);
        border-bottom: 2px solid #39FF14;
        padding: 15px 20px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 30px rgba(57, 255, 20, 0.1);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e14; }
    ::-webkit-scrollbar-thumb { background: #39FF1444; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #39FF1488; }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'Orbitron', monospace !important;
        color: #39FF14 !important;
        background: #131920 !important;
    }
    
    /* Divider */
    hr { border-color: #39FF1433 !important; }
    
    </style>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Header Component
# ──────────────────────────────────────────────

def render_header():
    """Render the top surveillance system header bar."""
    now = datetime.now()
    st.markdown(f"""
    <div class="header-bar">
        <div style="display: flex; align-items: center; gap: 15px;">
            <div style="font-family: 'Orbitron', monospace; font-size: 1.6rem; font-weight: 800; 
                        color: #39FF14; text-shadow: 0 0 20px rgba(57,255,20,0.5);">
                ◉ BORDER SENTINEL
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #6b7280; 
                        padding: 4px 10px; border: 1px solid #2a3a4a; border-radius: 4px;">
                EDGE AI SURVEILLANCE v2.0
            </div>
        </div>
        <div style="text-align: right;">
            <div style="font-family: 'Orbitron', monospace; font-size: 1.1rem; color: #39FF14;
                        text-shadow: 0 0 10px rgba(57,255,20,0.3);">
                {now.strftime("%H:%M:%S")}
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #6b7280;">
                {now.strftime("%Y-%m-%d")} | UTC+5:30
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Node Status Card
# ──────────────────────────────────────────────

def render_node_card(node_status: Dict, compact: bool = False):
    """Render a single node status card."""
    node_id = node_status.get("node_id", "??")
    state = node_status.get("state", "UNKNOWN")
    alert = node_status.get("alert_level", "NORMAL")
    battery = node_status.get("battery_percent", 100)
    temp = node_status.get("temperature_c", 25)
    inferences = node_status.get("total_inferences", 0)
    alerts_count = node_status.get("total_alerts", 0)
    
    # Determine card style
    if alert in ("HIGH_ALERT", "CRITICAL", "INTRUSION"):
        card_class = "alert-red"
        status_color = THEME["accent_red"]
        status_dot = "🔴"
    elif alert in ("SUSPICIOUS", "MODERATE"):
        card_class = "alert-amber"
        status_color = THEME["accent_amber"]
        status_dot = "🟡"
    else:
        card_class = ""
        status_color = THEME["accent_green"]
        status_dot = "🟢"
    
    # Battery icon
    if battery > 75:
        bat_icon = "🔋"
        bat_color = THEME["accent_green"]
    elif battery > 40:
        bat_icon = "🔋"
        bat_color = THEME["accent_amber"]
    else:
        bat_icon = "🪫"
        bat_color = THEME["accent_red"]
    
    st.markdown(f"""
    <div class="node-card {card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div style="font-family: 'Orbitron', monospace; font-size: 1rem; font-weight: 700; color: {status_color};">
                {status_dot} NODE-{node_id}
            </div>
            <span class="alert-badge alert-{'red' if alert in ('HIGH_ALERT','CRITICAL','INTRUSION') else 'amber' if alert in ('SUSPICIOUS','MODERATE') else 'green'}">
                {state}
            </span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
            <div style="color: #6b7280;">Battery</div>
            <div style="color: {bat_color}; text-align: right;">{bat_icon} {battery:.0f}%</div>
            <div style="color: #6b7280;">Temp</div>
            <div style="color: #c5c8c6; text-align: right;">{temp:.1f}°C</div>
            <div style="color: #6b7280;">Inferences</div>
            <div style="color: #c5c8c6; text-align: right;">{inferences}</div>
            <div style="color: #6b7280;">Alerts</div>
            <div style="color: {'#FF3131' if alerts_count > 0 else '#c5c8c6'}; text-align: right;">{alerts_count}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Waveform Plot
# ──────────────────────────────────────────────

def render_waveform(audio_data: np.ndarray, sample_rate: int = 22050):
    """Render live audio waveform using Plotly."""
    if audio_data is None or len(audio_data) == 0:
        audio_data = np.zeros(sample_rate)
    
    t = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
    
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=t, y=audio_data,
        mode='lines',
        line=dict(color=THEME["accent_green"], width=1),
        fill='tozeroy',
        fillcolor='rgba(57, 255, 20, 0.08)',
        name='Audio Signal'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=THEME["bg_primary"],
        plot_bgcolor=THEME["bg_secondary"],
        margin=dict(l=40, r=20, t=40, b=40),
        height=200,
        xaxis=dict(
            title='Time (s)', titlefont=dict(size=10, color=THEME["text_secondary"]),
            gridcolor=THEME["grid"], showgrid=True, gridwidth=1,
            range=[0, max(t) if len(t) > 0 else 1],
        ),
        yaxis=dict(
            title='Amplitude', titlefont=dict(size=10, color=THEME["text_secondary"]),
            gridcolor=THEME["grid"], showgrid=True, gridwidth=1,
            range=[-max(0.01, float(np.max(np.abs(audio_data))) * 1.2),
                    max(0.01, float(np.max(np.abs(audio_data))) * 1.2)],
        ),
        font=dict(family='JetBrains Mono, monospace', size=10, color=THEME["text_primary"]),
        showlegend=False,
    )
    
    return fig


# ──────────────────────────────────────────────
# Classification Bar Chart
# ──────────────────────────────────────────────

def render_classification_bars(predictions: Dict[str, float]):
    """Render real-time classification probability bars."""
    if not predictions:
        predictions = {"footstep": 0.0, "gunshot": 0.0, "noise": 0.0}
    
    labels = list(predictions.keys())
    values = list(predictions.values())
    
    colors = []
    for label, val in zip(labels, values):
        if "gunshot" in label.lower() or "balastic" in label.lower():
            colors.append(THEME["accent_red"] if val > 0.5 else "rgba(255, 49, 49, 0.4)")
        elif "footstep" in label.lower():
            colors.append(THEME["accent_amber"] if val > 0.7 else "rgba(255, 176, 0, 0.4)")
        else:
            colors.append(THEME["accent_green"] if val > 0.5 else "rgba(57, 255, 20, 0.4)")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values, y=[l.upper() for l in labels],
        orientation='h',
        marker=dict(color=colors, line=dict(width=1, color='rgba(255,255,255,0.1)')),
        text=[f"{v:.1%}" for v in values],
        textposition='inside',
        textfont=dict(family='Orbitron, monospace', size=14, color='white'),
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=THEME["bg_primary"],
        plot_bgcolor=THEME["bg_secondary"],
        margin=dict(l=10, r=20, t=10, b=10),
        height=180,
        xaxis=dict(
            range=[0, 1], gridcolor=THEME["grid"], showgrid=True,
            tickformat='.0%', tickfont=dict(size=9),
        ),
        yaxis=dict(tickfont=dict(family='Orbitron, monospace', size=12, color=THEME["text_primary"])),
        font=dict(family='JetBrains Mono, monospace', size=10, color=THEME["text_primary"]),
        showlegend=False,
        bargap=0.3,
    )
    
    return fig


# ──────────────────────────────────────────────
# Alert Panel
# ──────────────────────────────────────────────

def render_alert_panel(alerts: List[Dict]):
    """Render scrolling alert log panel."""
    if not alerts:
        st.markdown(
            '<div class="comm-log" style="color: #39FF14;">'
            '<div style="text-align: center; padding: 20px; color: #6b7280;">'
            '◉ SYSTEM NOMINAL — NO ALERTS'
            '</div></div>',
            unsafe_allow_html=True,
        )
        return
    
    rows = []
    for alert in reversed(alerts[-15:]):
        level = alert.get("threat_level", alert.get("alert_level", "NORMAL"))
        color = ALERT_COLORS.get(level, THEME["accent_green"])
        event = alert.get("event_type", alert.get("event", "unknown")).upper()
        conf = alert.get("confidence", 0)
        conf_pct = int(conf * 100)
        node_id = alert.get("node_id", "??")
        ts = alert.get("timestamp", "")
        
        if level in ("CRITICAL", "HIGH_ALERT", "HIGH"):
            icon = "&#x1F534;"   # 🔴
        elif level in ("MODERATE", "SUSPICIOUS"):
            icon = "&#x1F7E1;"   # 🟡
        else:
            icon = "&#x1F7E2;"   # 🟢
        
        rows.append(
            f'<div style="padding:6px 0;border-bottom:1px solid #1a2332;">'
            f'<span style="color:#6b7280;font-size:0.7rem;">{ts}</span> '
            f'<span style="color:{color};font-weight:700;">{icon} [{level}]</span> '
            f'<span style="color:#c5c8c6;">Node-{node_id} | {event} ({conf_pct}%)</span>'
            f'</div>'
        )
    
    body = "".join(rows)
    st.markdown(
        f'<div class="comm-log" style="max-height:300px;">{body}</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Communication Log
# ──────────────────────────────────────────────

def render_comm_log(logs: List[str]):
    """Render LoRa communication log panel."""
    if not logs:
        log_content = '<div style="text-align: center; color: #6b7280; padding: 20px;">Waiting for transmissions...</div>'
    else:
        log_content = ""
        for entry in logs[-20:]:
            if "ALERT" in entry or "⚠" in entry or "🚨" in entry:
                color = THEME["accent_red"]
            elif "Sending" in entry or "TX" in entry:
                color = THEME["accent_cyan"]
            elif "received" in entry.lower() or "BASE" in entry:
                color = THEME["accent_amber"]
            else:
                color = THEME["accent_green"]
            
            log_content += f'<div style="color: {color}; padding: 2px 0;">{entry}</div>'
    
    st.markdown(f"""
    <div class="comm-log">
        {log_content}
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Deployment Map
# ──────────────────────────────────────────────

def render_deployment_map(nodes: List[Dict]):
    """Render tactical deployment map showing node positions."""
    fig = go.Figure()
    
    # Grid background
    for i in range(11):
        fig.add_shape(type="line", x0=i, y0=0, x1=i, y1=10,
                      line=dict(color="rgba(57,255,20,0.08)", width=1))
        fig.add_shape(type="line", x0=0, y0=i, x1=10, y1=i,
                      line=dict(color="rgba(57,255,20,0.08)", width=1))
    
    # Base station (center)
    fig.add_trace(go.Scatter(
        x=[5], y=[5], mode='markers+text',
        marker=dict(size=25, color=THEME["accent_cyan"], symbol='diamond',
                    line=dict(width=2, color='white')),
        text=['BASE'], textposition='top center',
        textfont=dict(family='Orbitron', size=10, color=THEME["accent_cyan"]),
        name='Base Station'
    ))
    
    # Nodes
    node_positions = [(2, 8), (8, 7), (5, 2)]
    for i, node in enumerate(nodes):
        pos = node_positions[i] if i < len(node_positions) else (np.random.uniform(1, 9), np.random.uniform(1, 9))
        alert = node.get("alert_level", "NORMAL")
        
        color = ALERT_COLORS.get(alert, THEME["accent_green"])
        node_id = node.get("node_id", f"{i+1:02d}")
        
        # Connection line to base
        fig.add_trace(go.Scatter(
            x=[pos[0], 5], y=[pos[1], 5], mode='lines',
            line=dict(color=color, width=1, dash='dot'),
            showlegend=False, hoverinfo='skip'
        ))
        
        # Node marker
        fig.add_trace(go.Scatter(
            x=[pos[0]], y=[pos[1]], mode='markers+text',
            marker=dict(size=18, color=color, symbol='triangle-up',
                        line=dict(width=2, color='white')),
            text=[f'N-{node_id}'], textposition='top center',
            textfont=dict(family='Orbitron', size=9, color=color),
            name=f'Node {node_id}',
        ))
        
        # Signal radius
        fig.add_shape(type="circle",
            xref="x", yref="y",
            x0=pos[0]-1.5, y0=pos[1]-1.5, x1=pos[0]+1.5, y1=pos[1]+1.5,
            line=dict(color=color, width=1, dash="dot"),
            fillcolor=f"rgba({','.join([str(int(color[i:i+2], 16)) for i in (1,3,5)])}, 0.05)" if color.startswith('#') else "transparent",
        )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=THEME["bg_primary"],
        plot_bgcolor='#0d1318',
        margin=dict(l=10, r=10, t=10, b=10),
        height=350,
        xaxis=dict(range=[-0.5, 10.5], visible=False),
        yaxis=dict(range=[-0.5, 10.5], visible=False, scaleanchor="x"),
        showlegend=False,
        font=dict(family='JetBrains Mono, monospace'),
    )
    
    return fig


# ──────────────────────────────────────────────
# Signal Gauge
# ──────────────────────────────────────────────

def render_signal_gauge(rssi: float, snr: float):
    """Render signal strength indicator."""
    # RSSI: -30 (excellent) to -120 (no signal)
    strength = max(0, min(100, (rssi + 120) / 90 * 100))
    
    if strength > 70:
        color = THEME["accent_green"]
        label = "EXCELLENT"
    elif strength > 40:
        color = THEME["accent_amber"]
        label = "GOOD"
    else:
        color = THEME["accent_red"]
        label = "WEAK"
    
    bars = "▰" * int(strength / 10) + "▱" * (10 - int(strength / 10))
    
    st.markdown(f"""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
        <div style="color: #6b7280;">Signal: <span style="color: {color};">{bars}</span> <span style="color: {color};">{label}</span></div>
        <div style="color: #6b7280; font-size: 0.7rem;">RSSI: {rssi:.0f}dBm | SNR: {snr:.1f}dB</div>
    </div>
    """, unsafe_allow_html=True)
