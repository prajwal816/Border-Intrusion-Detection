"""
Audio Capture Server — runs in a subprocess where audio works.
Finds the best microphone, records continuously, saves frames to shared file.
"""
import sys
import time
import json
import numpy as np
import os

SAMPLE_RATE = 22050
FRAME_DURATION = 1.0
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)

def find_best_mic():
    """Try each input device and find the one that actually captures audio."""
    import sounddevice as sd
    
    devices = sd.query_devices()
    candidates = []
    
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            name = d['name'].lower()
            # Skip non-microphone devices
            if any(skip in name for skip in ['stereo mix', 'loopback', 'pc speaker', 'speaker', 'midi']):
                continue
            candidates.append((i, d['name'], d['default_samplerate']))
    
    print(f"[MIC] Testing {len(candidates)} microphone devices:", flush=True)
    
    best_device = None
    best_rms = 0
    
    for dev_id, name, native_sr in candidates:
        try:
            test_audio = sd.rec(
                int(0.5 * native_sr),
                samplerate=native_sr,
                channels=1,
                dtype='float32',
                device=dev_id
            )
            sd.wait()
            rms = float(np.sqrt(np.mean(test_audio**2)))
            # Skip invalid readings
            if np.isnan(rms) or np.isinf(rms) or rms > 10.0:
                print(f"  [{dev_id}] {name}: INVALID (rms={rms})", flush=True)
                continue
            print(f"  [{dev_id}] {name}: RMS={rms:.6f}", flush=True)
            if rms > best_rms:
                best_rms = rms
                best_device = (dev_id, name, native_sr)
        except Exception as e:
            print(f"  [{dev_id}] {name}: FAILED", flush=True)
    
    if best_device:
        print(f"[MIC] ✅ Selected: [{best_device[0]}] {best_device[1]} (RMS={best_rms:.6f})", flush=True)
    
    return best_device


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not output_path:
        print(json.dumps({"error": "No output path"}), flush=True)
        return
    
    output_path = os.path.abspath(output_path)
    parent = os.path.dirname(output_path)
    os.makedirs(parent, exist_ok=True)

    try:
        import sounddevice as sd
    except ImportError:
        print(json.dumps({"error": "sounddevice not available"}), flush=True)
        return

    mic = find_best_mic()
    if mic is None:
        print(json.dumps({"error": "No working microphone found"}), flush=True)
        return
    
    dev_id, dev_name, native_sr = mic
    use_resample = int(native_sr) != SAMPLE_RATE
    
    print(json.dumps({"status": "ready", "device": dev_name, "device_id": dev_id}), flush=True)

    while True:
        try:
            n_samples = int(FRAME_DURATION * native_sr)
            audio = sd.rec(n_samples, samplerate=native_sr, channels=1, dtype='float32', device=dev_id)
            sd.wait()
            frame = audio[:, 0].astype(np.float32)
            
            if use_resample:
                import librosa
                frame = librosa.resample(frame, orig_sr=native_sr, target_sr=SAMPLE_RATE)
            
            if len(frame) < FRAME_SIZE:
                frame = np.pad(frame, (0, FRAME_SIZE - len(frame)))
            else:
                frame = frame[:FRAME_SIZE]
            
            rms = float(np.sqrt(np.mean(frame**2)))
            peak = float(np.max(np.abs(frame)))
            print(f"[MIC] RMS={rms:.6f} Peak={peak:.4f}", flush=True)
            
            np.save(output_path, frame)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[MIC] Error: {e}", flush=True)
            time.sleep(1)

if __name__ == "__main__":
    main()
