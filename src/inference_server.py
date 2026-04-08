"""
Inference Server — runs in a subprocess where TF/ONNX DLLs work.
Communicates with Streamlit via stdin/stdout JSON.
Usage: started automatically by AudioClassifier.
"""
import sys
import json
import numpy as np
import os

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not model_path or not os.path.exists(model_path):
        print(json.dumps({"error": f"Model not found: {model_path}"}), flush=True)
        return

    # Load model based on extension
    session = None
    model_type = None

    if model_path.endswith(".onnx"):
        import onnxruntime as ort
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        inp_info = session.get_inputs()[0]
        input_name = inp_info.name
        model_type = "onnx"
    elif model_path.endswith(".tflite"):
        import tensorflow as tf
        session = tf.lite.Interpreter(model_path=model_path)
        session.allocate_tensors()
        model_type = "tflite"
    elif model_path.endswith(".h5"):
        import tensorflow as tf
        session = tf.keras.models.load_model(model_path, compile=False)
        model_type = "keras"

    # Signal ready
    print(json.dumps({"status": "ready", "model_type": model_type}), flush=True)

    # Inference loop — read requests from stdin, write results to stdout
    for line in sys.stdin:
        line = line.strip()
        if not line or line == "QUIT":
            break
        try:
            request = json.loads(line)
            features = np.array(request["features"], dtype=np.float32)

            if model_type == "onnx":
                outputs = session.run(None, {input_name: features})
                probs = outputs[0][0].tolist()
            elif model_type == "tflite":
                inp = session.get_input_details()
                out = session.get_output_details()
                session.set_tensor(inp[0]['index'], features)
                session.invoke()
                probs = session.get_tensor(out[0]['index'])[0].tolist()
            elif model_type == "keras":
                probs = session.predict(features, verbose=0)[0].tolist()
            else:
                probs = [0.33, 0.33, 0.34]

            print(json.dumps({"probs": probs}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    main()
