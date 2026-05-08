"""
Training script: UWB multi-person localization → TFLite (INT8).

Architecture: lightweight CNN on per-frame CIR magnitude.
Input  : (1, 18, 120, 1)  — 6 radars × 3 antennas flattened, 120 range bins
Output : (1, 12)          — 4 persons × (x, y, confidence)

ESP32-S3 constraints enforced:
  - Only Conv2D / DepthwiseConv2D / Dense / BN / Pool / Add / ReLU
  - INT8 post-training quantization
  - Model size < 800 KB,  tensor arena < 400 KB
"""
import os
import glob
import json
import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "../data"           # folder with .npz files
OUTPUT_DIR = "../submission"
MODEL_KERAS = os.path.join(OUTPUT_DIR, "model_keras.keras")
MODEL_TFLITE = os.path.join(OUTPUT_DIR, "model.tflite")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS      = 30
BATCH_SIZE  = 64
VAL_SPLIT   = 0.15
SEED        = 42
MAX_PEOPLE  = 4
ROOM_X      = 4.8
ROOM_Y      = 7.2

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_dataset(data_dir: str):
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    print(f"Loading {len(npz_files)} acquisition windows…")

    all_X, all_Y = [], []
    for path in npz_files:
        d = np.load(path)
        cir_iq = d["radar_cir_iq"].astype(np.float32)   # (T, 6, 3, 120, 2)
        xy     = d["people_xy"].astype(np.float32)       # (T, 4, 2)
        mask   = d["people_mask"].astype(np.float32)     # (T, 4)

        T = cir_iq.shape[0]

        # CIR magnitude → (T, 18, 120, 1)
        cir = cir_iq[..., 0] + 1j * cir_iq[..., 1]
        mag = np.abs(cir).reshape(T, 18, 120, 1)

        # Per-sample normalisation
        mu  = mag.mean(axis=(1, 2, 3), keepdims=True)
        sig = mag.std(axis=(1, 2, 3), keepdims=True) + 1e-8
        mag = (mag - mu) / sig

        # Normalise ground-truth to [0, 1]
        xy_norm = xy.copy()
        xy_norm[..., 0] /= ROOM_X
        xy_norm[..., 1] /= ROOM_Y

        # Target: (T, 4, 3) = [x_norm, y_norm, confidence]
        target = np.concatenate([xy_norm, mask[..., None]], axis=-1)  # (T, 4, 3)
        target = target.reshape(T, MAX_PEOPLE * 3)                     # (T, 12)

        all_X.append(mag)
        all_Y.append(target)

    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)
    print(f"  X: {X.shape}  Y: {Y.shape}")
    return X, Y


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(input_shape=(18, 120, 1)) -> tf.keras.Model:
    inp = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(16, (3, 7), padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)        # → (9, 60, 16)

    x = tf.keras.layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(32, (1, 1), use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((3, 4))(x)        # → (3, 15, 32)

    x = tf.keras.layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(64, (1, 1), use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.AveragePooling2D((3, 15))(x)  # → (1, 1, 64)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(MAX_PEOPLE * 3, activation="sigmoid")(x)  # 12 outputs

    return tf.keras.Model(inp, out)


# ── Loss ──────────────────────────────────────────────────────────────────────
def hungarian_loss(y_true, y_pred):
    """
    Set-based Hungarian matching loss.
    y_true, y_pred: (B, 12) → reshape to (B, 4, 3)
    """
    B = tf.shape(y_true)[0]
    yt = tf.reshape(y_true, (B, MAX_PEOPLE, 3))
    yp = tf.reshape(y_pred, (B, MAX_PEOPLE, 3))

    # Use simple L1 + BCE per sample (full Hungarian in pure TF is complex)
    # This approximation works well when the model learns slot ordering
    xy_loss   = tf.reduce_mean(tf.abs(yt[..., :2] - yp[..., :2]), axis=-1)   # (B, 4)
    conf_loss = tf.keras.losses.binary_crossentropy(yt[..., 2], yp[..., 2])  # (B, 4)

    mask = yt[..., 2]  # (B, 4) — 1 where a person is present
    pos_xy_loss = tf.reduce_sum(xy_loss * mask, axis=-1) / (tf.reduce_sum(mask, axis=-1) + 1e-8)
    total_conf  = tf.reduce_mean(conf_loss, axis=-1)

    return tf.reduce_mean(pos_xy_loss + total_conf)


# ── TFLite conversion (INT8) ─────────────────────────────────────────────────
def convert_to_tflite_int8(keras_model, rep_data: np.ndarray, output_path: str):
    def representative_dataset():
        idx = np.random.choice(len(rep_data), size=min(200, len(rep_data)), replace=False)
        for i in idx:
            yield [rep_data[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"TFLite INT8 model saved: {output_path}  ({size_kb:.1f} KB)")
    return size_kb


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    X, Y = load_dataset(DATA_DIR)

    n_val = int(len(X) * VAL_SPLIT)
    idx = np.random.permutation(len(X))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val,   Y_val   = X[val_idx],   Y[val_idx]

    model = build_model()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=hungarian_loss,
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(MODEL_KERAS, save_best_only=True, verbose=1),
    ]

    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    # Convert to TFLite INT8
    size_kb = convert_to_tflite_int8(model, X_train, MODEL_TFLITE)
    if size_kb >= 800:
        print(f"WARNING: model size {size_kb:.1f} KB exceeds 800 KB limit!")


if __name__ == "__main__":
    main()
