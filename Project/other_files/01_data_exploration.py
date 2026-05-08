"""
Data exploration script for the Multi-Radar UWB dataset.
Run this after downloading .npz files from Hugging Face.
Adjust DATA_DIR to point to the folder containing the .npz files.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DATA_DIR = "../data"   # adjust to where you downloaded the dataset

# ── Load one acquisition window ────────────────────────────────────────────
npz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
print(f"Found {len(npz_files)} acquisition windows")

data = np.load(npz_files[0])
print("\nKeys:", list(data.keys()))

radar_cir_iq = data["radar_cir_iq"]   # (T, 6, 3, 120, 2)
people_xy    = data["people_xy"]       # (T, 4, 2)
people_mask  = data["people_mask"]     # (T, 4)
timestamps   = data["timestamps"]      # (T,)

T = radar_cir_iq.shape[0]
print(f"\nframes      : {T}")
print(f"radar_cir_iq: {radar_cir_iq.shape}")
print(f"people_xy   : {people_xy.shape}")
print(f"people_mask : {people_mask.shape}")
print(f"timestamps  : {timestamps.shape}")

# Occupancy distribution across all files
print("\n── Occupancy distribution ──")
count_hist = np.zeros(5, dtype=int)
for path in npz_files:
    d = np.load(path)
    mask = d["people_mask"]          # (T, 4)
    occ = mask.sum(axis=1).astype(int)  # (T,)
    for n in occ:
        count_hist[n] += 1
for n, c in enumerate(count_hist):
    print(f"  {n} people: {c:6d} frames")

# ── CIR magnitude for one frame ─────────────────────────────────────────────
frame_idx = 100
cir = radar_cir_iq[frame_idx]        # (6, 3, 120, 2)
magnitude = np.sqrt(cir[..., 0]**2 + cir[..., 1]**2)  # (6, 3, 120)

fig, axes = plt.subplots(3, 6, figsize=(18, 6))
fig.suptitle(f"CIR magnitude — frame {frame_idx}")
for r in range(6):
    for a in range(3):
        axes[a, r].plot(magnitude[r, a])
        axes[a, r].set_title(f"R{r+1} A{a+1}", fontsize=8)
        axes[a, r].set_xlabel("bin")
        axes[a, r].set_ylim(0, None)
plt.tight_layout()
plt.savefig("cir_magnitude.png", dpi=100)
plt.close()
print("\nSaved cir_magnitude.png")

# ── Ground-truth trajectories ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 9))
ax.set_xlim(0, 4.8)
ax.set_ylim(0, 7.2)
ax.set_aspect("equal")
ax.set_title("Ground-truth positions (all frames, first window)")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
for p in range(4):
    valid = people_mask[:, p].astype(bool)
    if valid.sum() > 0:
        xy = people_xy[valid, p, :]
        ax.scatter(xy[:, 0], xy[:, 1], s=1, c=colors[p], label=f"person {p+1}")

# Draw radar positions (approximate from the diagram)
radar_positions = [
    (2.4, 0.0),   # SR250_1 (bottom)
    (4.8, 2.4),   # SR250_2 (right-lower)
    (4.8, 5.4),   # SR250_3 (right-upper)
    (2.4, 7.2),   # SR250_4 (top)
    (0.0, 5.4),   # SR250_5 (left-upper)
    (0.0, 2.4),   # SR250_6 (left-lower)
]
for i, (rx, ry) in enumerate(radar_positions):
    ax.scatter(rx, ry, marker="^", s=100, c="black", zorder=5)
    ax.annotate(f"R{i+1}", (rx, ry), textcoords="offset points", xytext=(5, 5), fontsize=8)

ax.legend(markerscale=5)
plt.tight_layout()
plt.savefig("trajectories.png", dpi=100)
plt.close()
print("Saved trajectories.png")

print("\nExploration done.")
