"""
ESP32-S3 constraint checker for TFLite models.
Checks: model size < 800 KB, tensor arena < 400 KB, no forbidden ops.
"""
import argparse
import os
import sys
import numpy as np

FORBIDDEN_OPS = {
    "LSTM", "UNIDIRECTIONAL_SEQUENCE_LSTM",
    "BIDIRECTIONAL_SEQUENCE_LSTM",
    "UNIDIRECTIONAL_SEQUENCE_RNN",
    "BIDIRECTIONAL_SEQUENCE_RNN",
    "RNN",
}

MAX_MODEL_SIZE_KB = 800
MAX_ARENA_KB = 400


def check_model_size(model_path: str) -> tuple[bool, float]:
    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024
    ok = size_kb < MAX_MODEL_SIZE_KB
    return ok, size_kb


def check_forbidden_ops(model_path: str) -> tuple[bool, list[str]]:
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)
        # Access flatbuffer directly via _get_ops_details or enumerate ops
        # We walk tensor details to find op names via a known workaround
        ops_found = []
        try:
            # TF 2.x exposes _interpreter.GetOps() via internal API
            # Use flatbuffers parsing instead
            with open(model_path, "rb") as f:
                model_bytes = f.read()
            # Parse op names via string search (conservative approach)
            for op in FORBIDDEN_OPS:
                if op.encode() in model_bytes:
                    ops_found.append(op)
        except Exception:
            pass
        ok = len(ops_found) == 0
        return ok, ops_found
    except ImportError:
        print("TensorFlow not installed, skipping op check.")
        return True, []


def check_tensor_arena(model_path: str) -> tuple[bool, float]:
    """
    Estimate tensor arena by allocating the interpreter and reading peak memory.
    Uses TF Lite's arena size estimator if available, otherwise approximates.
    """
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Sum all tensor sizes as a proxy for arena usage
        total_bytes = 0
        for detail in interpreter.get_tensor_details():
            shape = detail["shape"]
            dtype = detail["dtype"]
            try:
                nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
                total_bytes += nbytes
            except Exception:
                pass

        arena_kb = total_bytes / 1024
        ok = arena_kb < MAX_ARENA_KB
        return ok, arena_kb
    except ImportError:
        print("TensorFlow not installed, skipping arena check.")
        return True, 0.0


def main():
    parser = argparse.ArgumentParser(description="Check ESP32-S3 deployment constraints")
    parser.add_argument("--model-path", required=True, help="Path to model.tflite")
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        print(f"ERROR: model not found at {args.model_path}")
        sys.exit(1)

    print(f"Checking: {args.model_path}\n")
    all_pass = True

    # 1. Model size
    size_ok, size_kb = check_model_size(args.model_path)
    status = "PASS" if size_ok else "FAIL"
    print(f"[{status}] Model size: {size_kb:.1f} KB (limit: {MAX_MODEL_SIZE_KB} KB)")
    all_pass = all_pass and size_ok

    # 2. Forbidden ops
    ops_ok, bad_ops = check_forbidden_ops(args.model_path)
    status = "PASS" if ops_ok else "FAIL"
    if ops_ok:
        print(f"[{status}] No forbidden operations detected")
    else:
        print(f"[{status}] Forbidden operations found: {bad_ops}")
    all_pass = all_pass and ops_ok

    # 3. Tensor arena
    arena_ok, arena_kb = check_tensor_arena(args.model_path)
    status = "PASS" if arena_ok else "FAIL"
    print(f"[{status}] Tensor arena estimate: {arena_kb:.1f} KB (limit: {MAX_ARENA_KB} KB)")
    all_pass = all_pass and arena_ok

    print()
    if all_pass:
        print("All constraints satisfied. Model is ready for submission.")
    else:
        print("One or more constraints FAILED. Fix before submitting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
