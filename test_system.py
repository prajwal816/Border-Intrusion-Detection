"""Quick diagnostic test for the Border Sentinel system."""
import os, sys, time

print("=" * 50)
print("BORDER SENTINEL — Diagnostic Test")
print("=" * 50)

# 1. Test TFLite model loading
print("\n[1] Testing TFLite model...")
tflite_path = os.path.join(os.path.dirname(__file__), "models", "border_intrusion_model.tflite")
print(f"    Path: {tflite_path}")
print(f"    Exists: {os.path.exists(tflite_path)}")
if os.path.exists(tflite_path):
    print(f"    Size: {os.path.getsize(tflite_path)} bytes")
    try:
        t0 = time.time()
        import tensorflow as tf
        print(f"    TF version: {tf.__version__} (import took {time.time()-t0:.1f}s)")
        t0 = time.time()
        interp = tf.lite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        inp = interp.get_input_details()
        out = interp.get_output_details()
        print(f"    Input shape: {inp[0]['shape']}, dtype: {inp[0]['dtype']}")
        print(f"    Output shape: {out[0]['shape']}")
        print(f"    Load time: {time.time()-t0:.2f}s")
        print("    ✅ TFLite OK!")
    except Exception as e:
        print(f"    ❌ TFLite FAILED: {e}")

# 2. Test microphone
print("\n[2] Testing microphone...")
try:
    import sounddevice as sd
    print(f"    sounddevice available: True")
    devs = sd.query_devices()
    input_devs = [(i, d) for i, d in enumerate(devs) if d['max_input_channels'] > 0]
    print(f"    Input devices found: {len(input_devs)}")
    for i, d in input_devs:
        print(f"      [{i}] {d['name']} (channels={d['max_input_channels']}, sr={d['default_samplerate']})")
    print(f"    Default: {sd.default.device}")
    
    # Try a quick 0.5s capture
    print("    Attempting 0.5s capture...")
    import numpy as np
    audio = sd.rec(int(0.5 * 22050), samplerate=22050, channels=1, dtype='float32')
    sd.wait()
    rms = float(np.sqrt(np.mean(audio**2)))
    print(f"    Captured {len(audio)} samples, RMS={rms:.6f}")
    if rms > 0.001:
        print("    ✅ Microphone is capturing audio!")
    else:
        print("    ⚠️ Very low signal - mic may be muted or not working")
except ImportError:
    print("    ❌ sounddevice not installed")
except Exception as e:
    print(f"    ❌ Microphone error: {e}")

# 3. Test librosa
print("\n[3] Testing librosa...")
try:
    import librosa
    print(f"    librosa version: {librosa.__version__}")
    print("    ✅ librosa OK!")
except Exception as e:
    print(f"    ❌ librosa error: {e}")

print("\n" + "=" * 50)
print("Diagnostic complete!")
