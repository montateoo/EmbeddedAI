import argparse
import json
import os
import numpy as np
import tensorflow as tf


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.tflite")

CONFIDENCE_THRESHOLD = 0.5
ROOM_X = 4.8
ROOM_Y = 7.2


def preprocess(radar_cir_iq: np.ndarray) -> np.ndarray:
    """
    radar_cir_iq: (T, 6, 3, 120, 2)
    Returns: (T, H, W, C) float32 ready for the TFLite model.
    Adapt this to match whatever preprocessing was used during training.
    """
    T = radar_cir_iq.shape[0]
    # Compute magnitude: (T, 6, 3, 120)
    cir = radar_cir_iq[..., 0] + 1j * radar_cir_iq[..., 1]
    magnitude = np.abs(cir).astype(np.float32)

    # Flatten radar/antenna/bin dims -> (T, 6*3, 120) then reshape for Conv2D
    # Shape: (T, 18, 120, 1)
    magnitude = magnitude.reshape(T, 18, 120, 1)

    # Normalize per-sample
    mean = magnitude.mean(axis=(1, 2, 3), keepdims=True)
    std = magnitude.std(axis=(1, 2, 3), keepdims=True) + 1e-8
    magnitude = (magnitude - mean) / std

    return magnitude


def run_inference(model_path: str, data: np.ndarray):
    """Run TFLite inference frame-by-frame, return list of raw outputs."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    results = []
    for i in range(data.shape[0]):
        frame = data[i:i+1]  # (1, H, W, C)

        # Handle quantized input
        if input_details[0]["dtype"] == np.int8:
            scale, zero_point = input_details[0]["quantization"]
            frame = (frame / scale + zero_point).astype(np.int8)

        interpreter.set_tensor(input_details[0]["index"], frame)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]["index"])

        # Handle quantized output
        if output_details[0]["dtype"] == np.int8:
            scale, zero_point = output_details[0]["quantization"]
            output = (output.astype(np.float32) - zero_point) * scale

        results.append(output[0])  # remove batch dim

    return results


def decode_output(raw_output: np.ndarray) -> list[list[float]]:
    """
    Decode raw model output to a list of [x, y] positions.
    Expected output shape: (4, 3) -> [x, y, confidence] for up to 4 people.
    Adapt this to your actual model output format.
    """
    localizations = []
    # Assume output is (4, 3): x, y, confidence per slot
    if raw_output.ndim == 1:
        # Reshape if flat: 4 people * 3 values
        if raw_output.shape[0] == 12:
            raw_output = raw_output.reshape(4, 3)
        else:
            return localizations

    for i in range(raw_output.shape[0]):
        x, y, conf = float(raw_output[i, 0]), float(raw_output[i, 1]), float(raw_output[i, 2])
        if conf >= CONFIDENCE_THRESHOLD:
            # Clip to room bounds
            x = float(np.clip(x, 0.0, ROOM_X))
            y = float(np.clip(y, 0.0, ROOM_Y))
            localizations.append([round(x, 3), round(y, 3)])

    return localizations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True, help="Path to input .npy file (T, 6, 3, 120, 2)")
    parser.add_argument("--output-path", required=True, help="Path to output .jsonl file")
    args = parser.parse_args()

    radar_cir_iq = np.load(args.input_path)  # (T, 6, 3, 120, 2)
    T = radar_cir_iq.shape[0]

    preprocessed = preprocess(radar_cir_iq)          # (T, 18, 120, 1)
    raw_outputs = run_inference(MODEL_PATH, preprocessed)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        for frame_idx, raw_out in enumerate(raw_outputs):
            localizations = decode_output(raw_out)
            record = {"frame": frame_idx, "localizations": localizations}
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {T} frames to {args.output_path}")


if __name__ == "__main__":
    main()
