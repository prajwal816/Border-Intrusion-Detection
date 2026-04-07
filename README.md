# 🛡️ Border Sentinel — Edge AI Intrusion Detection System

<div align="center">

**Real-time Acoustic Classification for Border Surveillance**

*ESP32 + MEMS Microphone + LoRa — Simulated Entirely in Software*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 🎯 Overview

**Border Sentinel** is a production-level software system that simulates an Edge AI-based border intrusion detection network. It processes real-time audio input from a laptop microphone and classifies sounds into three categories:

| Class | Description | Alert Trigger |
|---|---|---|
| 🚶 **Footstep** | Human footstep sounds | Intrusion Detected (>85% confidence) |
| 💥 **Gunshot** | Firearm discharge sounds | High Alert (>70% confidence) |
| 🌿 **Noise** | Environmental sounds (rain, birds, wind, etc.) | Normal |

The system simulates a complete hardware deployment:
- **ESP32 microcontrollers** with MEMS microphones at sensor nodes
- **TinyML inference** running on-edge for acoustic classification
- **LoRa radio** communication between nodes and a base station
- **Military-grade surveillance dashboard** for real-time monitoring

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                    BASE STATION                       │
│  ┌─────────────┐  ┌──────────┐  ┌─────────────────┐ │
│  │  LoRa RX    │←─│ Decision │←─│   Dashboard     │ │
│  │  Receiver   │  │  Engine  │  │   (Streamlit)   │ │
│  └──────┬──────┘  └──────────┘  └─────────────────┘ │
└─────────┼────────────────────────────────────────────┘
          │ LoRa Radio (Simulated)
    ┌─────┼──────┬──────────────┐
    │     │      │              │
┌───▼──┐ ┌▼────┐ ┌▼─────┐
│NODE-01│ │N-02 │ │N-03  │    Edge Sensor Nodes
│ESP32  │ │ESP32│ │ESP32 │    (VirtualESP32Node)
│MEMS   │ │MEMS │ │MEMS  │
│TinyML │ │TinyML│ │TinyML│
└───────┘ └─────┘ └──────┘
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| **Test Accuracy** | 94.08% |
| **Best Val Accuracy** | 94.67% |
| **Model Parameters** | 423,555 |
| **TFLite Size** | 427 KB |
| **Architecture** | 4-block CNN (Conv2D → BN → ReLU → MaxPool) |
| **Features** | 40 MFCCs, 128 Mel bands |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (optional — pre-trained model included)

```bash
python -m src.train_model
```

### 3. Launch the Dashboard

```bash
streamlit run app.py
```

### 4. Start Detection

1. Open the dashboard in your browser (http://localhost:8501)
2. Click **▶ START** in the sidebar
3. The system will begin processing audio and displaying results

---

## 📁 Project Structure

```
Border-Intrusion-Detection/
├── .streamlit/
│   └── config.toml              # Dark military theme
├── dataset/
│   ├── balastic/                 # Gunshot samples (374 files)
│   ├── footsteps/                # Footstep samples (80 files)
│   └── noise/                    # Environmental noise (208 files)
├── models/
│   ├── border_intrusion_model.h5      # Keras model
│   ├── border_intrusion_model.tflite  # Edge-optimized TFLite
│   └── training_metadata.json         # Training metrics
├── notebooks/
│   └── eda.ipynb                 # Exploratory Data Analysis
├── results/
│   ├── training_curves.png       # Accuracy/Loss plots
│   └── confusion_matrix.png      # Confusion matrix
├── src/
│   ├── __init__.py
│   ├── audio.py                  # Audio capture & MFCC extraction
│   ├── model.py                  # ML model loading & inference
│   ├── node.py                   # VirtualESP32Node simulation
│   ├── communication.py          # LoRa TX/RX + Base Station
│   ├── decision.py               # Alert decision engine
│   ├── train_model.py            # Model training pipeline
│   └── ui_components.py          # Dashboard UI components
├── app.py                        # Main Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🧩 Module Details

### `audio.py` — Audio Input Layer
- Real-time microphone capture via `sounddevice`
- Fallback replay mode using dataset samples
- MFCC and Mel-spectrogram feature extraction
- Energy-based wake-on-sound detection

### `model.py` — AI Model Integration
- Supports Keras (.h5) and TFLite (.tflite) inference
- Fallback heuristic classifier when no model is available
- Inference timing and performance metrics

### `node.py` — Hardware Abstraction Layer
- `VirtualESP32Node` simulating ESP32 behavior
- State machine: BOOT → IDLE → LISTENING → CAPTURING → PROCESSING → TRANSMITTING
- Simulated battery drain, temperature, CPU/memory usage
- Watchdog timer and wake-on-sound logic

### `communication.py` — LoRa Simulation
- `LoRaTransmitter`: Packet formation, RSSI/SNR simulation, CRC
- `BaseStationReceiver`: Multi-node packet reception, alert aggregation
- Realistic radio parameters (SF7, BW125, 915MHz ISM)

### `decision.py` — Decision Engine
- Threshold-based alert classification
- Consecutive detection escalation
- Alert cooldown to prevent spamming
- 5 threat levels: CLEAR → LOW → MODERATE → HIGH → CRITICAL

---

## 🖥️ Dashboard Features

| Panel | Description |
|---|---|
| **Node Network Status** | Status cards for 3 nodes with battery, temp, inference count |
| **Live Audio Waveform** | Real-time scrolling waveform visualization |
| **Classification Output** | Horizontal bar chart with color-coded probabilities |
| **Alert Panel** | Scrolling log of intrusion alerts with timestamps |
| **Deployment Map** | Tactical grid map with node positions and signal radius |
| **Communication Log** | Live LoRa TX/RX packet logs |
| **Signal Quality** | RSSI, SNR, and latency indicators |

---

## ⚙️ Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | TensorFlow / TFLite |
| Audio Processing | Librosa, SoundDevice |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Signal Processing | NumPy, SciPy |

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
