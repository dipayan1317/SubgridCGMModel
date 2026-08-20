import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LogNorm
from tqdm import tqdm

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
nx = 8
ny = 16
dtype = np.float64

folder_name = "pdf_trial_63/"

cool_filename = folder_name + "cool_rate.bin"
rho_filename  = folder_name + "rho.bin"
pdf_filename  = folder_name + "pdf.bin"

# PDF parameters
pdf_dtype = np.float32
n_pdf_bins = 40

T_min = 1e3
T_max = 1e7

# Number of snapshots to render
n_selected = 100

# Small floors for logarithmic scales
eps = 1e-30
pdf_floor = 1e-30


# ============================================================
# TEMPERATURE BINS
# ============================================================

T_edges = np.logspace(
    np.log10(T_min),
    np.log10(T_max),
    n_pdf_bins + 1
)

T_centers = 0.5 * (
    T_edges[:-1] + T_edges[1:]
)

logT_centers = np.log10(T_centers)


# ============================================================
# READ BINARY FILES
# ============================================================

cool_data = np.fromfile(
    cool_filename,
    dtype=dtype
)

rho_data = np.fromfile(
    rho_filename,
    dtype=dtype
)

pdf_data = np.fromfile(
    pdf_filename,
    dtype=pdf_dtype
)


# ============================================================
# NUMBER OF SNAPSHOTS
# ============================================================

cells = nx * ny

nt_cool = cool_data.size // cells
nt_rho = rho_data.size // cells

pdf_values_per_snapshot = n_pdf_bins * nx * ny

nt_pdf = pdf_data.size // pdf_values_per_snapshot

nt = min(
    nt_cool,
    nt_rho,
    nt_pdf
)

print(f"Cooling snapshots : {nt_cool}")
print(f"Density snapshots : {nt_rho}")
print(f"PDF snapshots     : {nt_pdf}")
print(f"Using             : {nt} snapshots")


# ============================================================
# SELECT 100 UNIFORMLY SPACED SNAPSHOTS
# ============================================================

n_selected = min(
    n_selected,
    nt
)

selected_frames = np.linspace(
    0,
    nt - 1,
    n_selected,
    dtype=int
)

print(
    f"Rendering {len(selected_frames)} "
    f"uniformly spaced snapshots"
)

print(
    f"First frame : {selected_frames[0]}"
)

print(
    f"Last frame  : {selected_frames[-1]}"
)


# ============================================================
# RESHAPE COOLING RATE
# ============================================================

cool_rate = cool_data[
    :nt * ny * nx
].reshape(
    nt,
    ny,
    nx
)


# ============================================================
# RESHAPE DENSITY
# ============================================================

rho = rho_data[
    :nt * ny * nx
].reshape(
    nt,
    ny,
    nx
)


# ============================================================
# RESHAPE PDF
#
# PDF file contains:
#
#     (snapshot, temperature, x, y)
#
# Convert to:
#
#     (snapshot, temperature, y, x)
# ============================================================

pdf = pdf_data[
    :nt * pdf_values_per_snapshot
].reshape(
    nt,
    n_pdf_bins,
    nx,
    ny
)

pdf = np.transpose(
    pdf,
    axes=(0, 1, 3, 2)
)


# ============================================================
# DENSITY FOR LOGARITHMIC DISPLAY
# ============================================================

rho = np.abs(rho)

print("Cooling shape :", cool_rate.shape)
print("Density shape :", rho.shape)
print("PDF shape     :", pdf.shape)


# ============================================================
# PDF LOG-SCALE LIMITS
# ============================================================

positive_pdf = pdf[pdf > 0]

if positive_pdf.size > 0:
    pdf_min_positive = positive_pdf.min()
else:
    pdf_min_positive = pdf_floor

pdf_floor = max(
    pdf_min_positive * 0.1,
    1e-30
)

pdf_ymax = np.max(pdf) * 1.05

print(
    f"PDF minimum positive value : "
    f"{pdf_min_positive:.3e}"
)

print(
    f"PDF plotting floor         : "
    f"{pdf_floor:.3e}"
)

print(
    f"PDF maximum                : "
    f"{pdf_ymax:.3e}"
)


# ============================================================
# FIRST PDF SNAPSHOT
# ============================================================

fig_pdf, pdf_axes = plt.subplots(
    ny,
    nx,
    figsize=(16, 24),
    sharex=True,
    sharey=True
)

pdf_axes = np.asarray(
    pdf_axes
).reshape(
    ny,
    nx
)

first_frame = selected_frames[0]


# ------------------------------------------------------------
# Plot first PDF snapshot
# ------------------------------------------------------------

for iy in range(ny):

    for ix in range(nx):

        ax = pdf_axes[iy, ix]

        pdf_curve = np.maximum(
            pdf[first_frame, :, iy, ix],
            pdf_floor
        )

        ax.plot(
            logT_centers,
            pdf_curve,
            linewidth=1.2
        )

        # Logarithmic PDF axis
        ax.set_yscale("log")

        ax.set_xlim(
            np.log10(T_min),
            np.log10(T_max)
        )

        ax.set_ylim(
            pdf_floor,
            pdf_ymax
        )

        ax.grid(
            True,
            alpha=0.25,
            which="both"
        )

        ax.text(
            0.05,
            0.90,
            f"({ix}, {iy})",
            transform=ax.transAxes,
            fontsize=8
        )


# ------------------------------------------------------------
# Axis labels
# ------------------------------------------------------------

for iy in range(ny):

    pdf_axes[iy, 0].set_ylabel(
        "PDF",
        fontsize=8
    )

for ix in range(nx):

    pdf_axes[-1, ix].set_xlabel(
        r"$\log_{10}(T/\mathrm{K})$",
        fontsize=8
    )


fig_pdf.suptitle(
    f"Temperature PDFs — Snapshot {first_frame}",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "pdf_grid_snapshot_0.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig_pdf)

print(
    "Saved pdf_grid_snapshot_0.png"
)


# ============================================================
# COMBINED ANIMATION FIGURE
# ============================================================

fig = plt.figure(
    figsize=(20, 25)
)


# ============================================================
# COOLING RATE
# ============================================================

ax_cool = fig.add_axes(
    [0.05, 0.72, 0.38, 0.23]
)

cool_im = ax_cool.imshow(
    cool_rate[first_frame],
    origin="lower",
    cmap="viridis",
    vmin=cool_rate[first_frame].min(),
    vmax=cool_rate[first_frame].max()
)

ax_cool.set_title(
    "Cooling Rate"
)

ax_cool.set_xlabel("x")
ax_cool.set_ylabel("y")

cool_cbar = fig.colorbar(
    cool_im,
    ax=ax_cool
)

cool_cbar.set_label(
    "Cooling Rate"
)


# ============================================================
# DENSITY
# ============================================================

ax_rho = fig.add_axes(
    [0.57, 0.72, 0.38, 0.23]
)

rho_im = ax_rho.imshow(
    rho[first_frame] + eps,
    origin="lower",
    cmap="plasma",
    norm=LogNorm(
        vmin=max(
            rho[first_frame].min(),
            eps
        ),
        vmax=max(
            rho[first_frame].max(),
            eps
        )
    )
)

ax_rho.set_title(
    "Density"
)

ax_rho.set_xlabel("x")
ax_rho.set_ylabel("y")

rho_cbar = fig.colorbar(
    rho_im,
    ax=ax_rho
)

rho_cbar.set_label(
    "Density"
)


# ============================================================
# 16 × 8 PDF GRID
# ============================================================

pdf_lines = np.empty(
    (ny, nx),
    dtype=object
)


# ------------------------------------------------------------
# Grid position
# ------------------------------------------------------------

left = 0.055
bottom = 0.035

total_width = 0.89
total_height = 0.62

gap_x = 0.008
gap_y = 0.006

cell_width = (
    total_width
    - (nx - 1) * gap_x
) / nx

cell_height = (
    total_height
    - (ny - 1) * gap_y
) / ny


# ------------------------------------------------------------
# Create PDF axes and curves
#
# IMPORTANT:
# (0,0) is placed at the BOTTOM-LEFT.
# ------------------------------------------------------------

for iy in range(ny):

    for ix in range(nx):

        x = (
            left
            + ix * (
                cell_width
                + gap_x
            )
        )

        # No (ny - 1 - iy) here.
        # Therefore iy=0 is at the bottom.
        y = (
            bottom
            + iy * (
                cell_height
                + gap_y
            )
        )

        ax = fig.add_axes(
            [
                x,
                y,
                cell_width,
                cell_height
            ]
        )

        pdf_curve = np.maximum(
            pdf[first_frame, :, iy, ix],
            pdf_floor
        )

        line, = ax.plot(
            logT_centers,
            pdf_curve,
            linewidth=0.9
        )

        pdf_lines[iy, ix] = line

        # Logarithmic PDF y-axis
        ax.set_yscale("log")

        ax.set_xlim(
            np.log10(T_min),
            np.log10(T_max)
        )

        ax.set_ylim(
            pdf_floor,
            pdf_ymax
        )

        ax.grid(
            True,
            alpha=0.2,
            which="both"
        )

        # Cell index
        ax.text(
            0.05,
            0.80,
            f"{ix},{iy}",
            transform=ax.transAxes,
            fontsize=5
        )

        # Only show y labels on first column
        if ix != 0:
            ax.set_yticklabels([])

        # Only show x labels on bottom row
        if iy != 0:
            ax.set_xticklabels([])


# ============================================================
# GLOBAL PDF LABELS
# ============================================================

fig.text(
    0.50,
    0.015,
    r"$\log_{10}(T/\mathrm{K})$",
    ha="center",
    fontsize=14
)

fig.text(
    0.015,
    0.35,
    "PDF",
    va="center",
    rotation="vertical",
    fontsize=14
)


# ============================================================
# OVERALL TITLE
# ============================================================

suptitle = fig.suptitle(
    f"Snapshot {first_frame}",
    fontsize=18
)


# ============================================================
# UPDATE FUNCTION
# ============================================================

def update(frame):

    # --------------------------------------------------------
    # Cooling rate - LINEAR
    # --------------------------------------------------------

    cool_frame = cool_rate[frame]

    cool_im.set_array(
        cool_frame
    )

    cool_im.set_clim(
        vmin=cool_frame.min(),
        vmax=cool_frame.max()
    )

    cool_cbar.update_normal(
        cool_im
    )


    # --------------------------------------------------------
    # Density - LOG
    # --------------------------------------------------------

    rho_frame = rho[frame] + eps

    rho_im.set_array(
        rho_frame
    )

    rho_im.set_norm(
        LogNorm(
            vmin=max(
                rho_frame.min(),
                eps
            ),
            vmax=max(
                rho_frame.max(),
                eps
            )
        )
    )

    rho_cbar.update_normal(
        rho_im
    )


    # --------------------------------------------------------
    # Update all 128 individual PDFs
    # --------------------------------------------------------

    for iy in range(ny):

        for ix in range(nx):

            pdf_curve = np.maximum(
                pdf[frame, :, iy, ix],
                pdf_floor
            )

            pdf_lines[iy, ix].set_ydata(
                pdf_curve
            )


    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    suptitle.set_text(
        f"Snapshot {frame}"
    )


    return (
        [cool_im, rho_im]
        + list(pdf_lines.ravel())
        + [suptitle]
    )


# ============================================================
# CREATE ANIMATION WITH TQDM
# ============================================================

progress = tqdm(
    total=len(selected_frames),
    desc="Rendering MP4",
    unit="frame"
)


def update_with_progress(frame):

    result = update(frame)

    progress.update(1)

    return result


# ============================================================
# ANIMATION
# ============================================================

anim = FuncAnimation(
    fig,
    update_with_progress,
    frames=selected_frames,
    interval=100,
    blit=False
)


# ============================================================
# FFMPEG WRITER
# ============================================================

writer = FFMpegWriter(
    fps=10,
    bitrate=5000
)


# ============================================================
# SAVE MP4
# ============================================================

try:

    anim.save(
        "pdf_grid_evolution.mp4",
        writer=writer,
        dpi=150
    )

finally:

    progress.close()


# ============================================================
# CLOSE
# ============================================================

plt.close(fig)

print(
    "Saved pdf_grid_evolution.mp4"
)