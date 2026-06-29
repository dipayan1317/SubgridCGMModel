import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
nx = 8          # Change to your x dimension
ny = 16          # Change to your y dimension
dtype = np.float64
filename = "gate16_8/cool_rate.bin"

# ------------------------------------------------------------
# Read binary file
# ------------------------------------------------------------
data = np.fromfile(filename, dtype=dtype)

nt = data.size // (nx * ny)

cool_rate = data.reshape(nt, ny, nx)

print(f"Loaded {nt} snapshots.")

# ------------------------------------------------------------
# Plot first snapshot
# ------------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.imshow(cool_rate[0], origin="lower", cmap="viridis")
plt.colorbar(label="Cooling Rate")
plt.title("Cooling Rate (Snapshot 0)")
plt.tight_layout()
plt.savefig("cool_rate_snapshot_0.png", dpi=300)
plt.close()
print("Saved cool_rate_snapshot_0.png")

# ------------------------------------------------------------
# Create GIF
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))

# Use fixed color scale over all frames
vmin = cool_rate.min()
vmax = cool_rate.max()

im = ax.imshow(
    cool_rate[0],
    origin="lower",
    cmap="viridis",
    vmin=vmin,
    vmax=vmax
)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Cooling Rate")

title = ax.set_title("Snapshot 0")

def update(frame):
    im.set_array(cool_rate[frame])
    title.set_text(f"Snapshot {frame}")
    return im, title

anim = FuncAnimation(
    fig,
    update,
    frames=nt,
    interval=100,   # milliseconds/frame
    blit=False
)

anim.save(
    "cool_rate_evolution.gif",
    writer=PillowWriter(fps=10)
)

plt.close(fig)

print("Saved cool_rate_evolution.gif")