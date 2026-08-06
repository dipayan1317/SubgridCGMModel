# ============================================================================
# Plot actual and predicted temperature PDFs
# Merged version (single-threaded)
# - Keeps original folder paths
# - Adds gate extraction
# - Adds quantitative metrics
# - Adds improved diagnostics
# - NO multiprocessing / parallel rendering
# ============================================================================

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors

import torch

from scipy.stats import pearsonr
from tqdm import tqdm

# ============================================================================
# PATHS
# ============================================================================

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from conv_nn.pdf_cnn import (
    snapshot_pred,
    snapshot_pred_with_gate,
    lambda_cool,
    compute_cooling_rate,
)

from data_preprocess import simulation_data

# ============================================================================
# DEVICE
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# RUN TOGGLES
# ============================================================================

RUN_PDF_ANIMATION = False

RUN_PDF_COMPARE_ANIMATION = False

RUN_COOLING_SCATTER = True

RUN_COOLING_HISTOGRAM = True

RUN_COOLING_COMPARE_ANIMATION = True

RUN_DENSITY_GATE_ANIMATION = False

RUN_FOURWAY_COMPARE_ANIMATION = False

# ============================================================================
# SETTINGS
# ============================================================================

resolution = (512, 256)

downsample = 32

bins = 40

folder_path = (
    f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/"
    f"AthenaK_legacy/datafiles/sct{resolution}_32"
)

PDF_MOCKS_DIR = "mocks/pdf"
MODEL_SAVE_DIR = f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/conv_nn/pdf_model_saves"

os.makedirs(PDF_MOCKS_DIR, exist_ok=True)

gif_path = os.path.join(PDF_MOCKS_DIR, "pdf_animation.gif")

gif_compare = os.path.join(
    PDF_MOCKS_DIR,
    "pdf_compare_animation.gif",
)

gif_cooling = os.path.join(
    PDF_MOCKS_DIR,
    "pdf_cooling_compare.gif",
)

first_frame_path = os.path.join(
    PDF_MOCKS_DIR,
    "pdf_snapshot_t0.png",
)

temp_frame_path = os.path.join(
    PDF_MOCKS_DIR,
    "temp_snapshot_t0.png",
)

# ============================================================================
# METRICS
# ============================================================================

def print_metrics(true, pred, label):

    """
    Log-space bias
    Log-space RMSE
    Pearson correlation
    """

    mask = (true > 0) & (pred > 0)

    print()

    print("=" * 70)

    print(label)

    print("=" * 70)

    print(f"Pixels used : {mask.sum()} / {true.size}")

    if mask.sum() < 2:

        print("Not enough positive pixels.")

        return

    log_true = np.log10(true[mask])

    log_pred = np.log10(pred[mask])

    bias = np.mean(log_pred - log_true)

    rmse = np.sqrt(np.mean((log_pred - log_true) ** 2))

    corr, _ = pearsonr(log_true, log_pred)

    print(f"Log Bias : {bias:+.4f} dex")

    print(f"Log RMSE : {rmse:.4f} dex")

    print(f"Pearson  : {corr:.5f}")


# ============================================================================
# TEMPERATURE BINS
# ============================================================================

temp_bins = np.logspace(
    3.0,
    7.0,
    bins + 1,
)

temp_centers = np.sqrt(
    temp_bins[:-1] *
    temp_bins[1:]
)

log_temp_centers = 0.5 * (
    np.log10(temp_bins[:-1]) +
    np.log10(temp_bins[1:])
)

active_bin_start = np.searchsorted(
    temp_centers,
    10 ** 4.5,
)

active_bin_end = np.searchsorted(
    temp_centers,
    10 ** 5.5,
)

cmap = plt.get_cmap("inferno")

norm = colors.Normalize(
    vmin=3.0,
    vmax=7.0,
)

# ============================================================================
# BATCH CNN PREDICTION
# ============================================================================

def batch_predict_with_gate(
    sim_data,
    downsample,
    resolution,
    device,
):

    """
    Predict every snapshot once.

    Returns
    -------
    conv_temp_pdf
    gate_maps
    vort_maps
    """

    from conv_nn.pdf_cnn import (
        ConvNN,
        in_channels,
        out_channels,
        kernel_size,
        layer_size1,
        layer_size2,
        layer_size3,
        layer_size4,
    )

    model_path = os.path.join(
        MODEL_SAVE_DIR,
        f"cnn_{resolution}_{downsample}.pth",
    )

    cnn_model = ConvNN(
        in_channels,
        layer_size1,
        layer_size2,
        layer_size3,
        layer_size4,
        out_channels,
        kernel_size,
    ).to(device)

    cnn_model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
        )
    )

    cnn_model.eval()

    input_mean = torch.tensor(
        np.load(
            os.path.join(
                MODEL_SAVE_DIR,
                f"cnn_{resolution}_{downsample}_input_mean.npy",
            )
        ),
        dtype=torch.float32,
        device=device,
    )

    input_std = torch.tensor(
        np.load(
            os.path.join(
                MODEL_SAVE_DIR,
                f"cnn_{resolution}_{downsample}_input_std.npy",
            )
        ),
        dtype=torch.float32,
        device=device,
    )

    nt = sim_data.rho.shape[0]

    nx = resolution[0] // downsample

    ny = resolution[1] // downsample

    conv_temp_pdf = np.zeros(
        (
            nt,
            out_channels,
            nx,
            ny,
        ),
        dtype=np.float32,
    )

    gate_maps = np.zeros(
        (
            nt,
            nx,
            ny,
        ),
        dtype=np.float32,
    )

    vort_maps = np.zeros_like(
        gate_maps
    )

    print()

    print("Running CNN prediction...")

    with torch.no_grad():

        for i in tqdm(range(nt)):

            rho = sim_data.coarse_grain(
                sim_data.rho[i]
            )

            temp = sim_data.coarse_grain(
                sim_data.temp[i]
            )

            ux = sim_data.coarse_grain(
                sim_data.ux[i]
            )

            uy = sim_data.coarse_grain(
                sim_data.uy[i]
            )

            ps = sim_data.coarse_grain(
                sim_data.ps[i]
            )

            x = np.stack(
                [
                    rho,
                    temp,
                    ux,
                    uy,
                    ps,
                ],
                axis=0,
            ).astype(np.float32)

            x = torch.from_numpy(
                x
            ).unsqueeze(0).to(device)

            x = (
                x - input_mean
            ) / input_std

            enriched = cnn_model.mixing(x)

            mixing = enriched[
                :,
                -cnn_model._N_MIXING:,
                :,
                :
            ]

            gate = cnn_model.gate_branch(
                mixing
            )

            features = cnn_model.encoder(
                enriched
            )

            logits = cnn_model.decoder(
                features
            )

            pdf = cnn_model.pdf_activation(
                logits,
                gate,
            )

            conv_temp_pdf[i] = (
                pdf[0]
                .cpu()
                .numpy()
            )

            gate_maps[i] = (
                gate[0, 0]
                .cpu()
                .numpy()
            )

            vort_maps[i] = (
                mixing[0, 0]
                .cpu()
                .numpy()
            )

    conv_temp_pdf /= (
        conv_temp_pdf.sum(
            axis=1,
            keepdims=True,
        ) + 1e-12
    )

    return (
        conv_temp_pdf,
        gate_maps,
        vort_maps,
    )

# ============================================================================
# LOAD DATA
# ============================================================================

print()
print("=" * 70)
print("Loading simulation data")
print("=" * 70)

sim_data = simulation_data()

sim_data.down_sample = downsample
sim_data.resolution = resolution

sim_data.rho = np.load(f"{folder_path}/rho.npy")
sim_data.temp = np.load(f"{folder_path}/temp.npy")
sim_data.pressure = np.load(f"{folder_path}/pressure.npy")
sim_data.ux = np.load(f"{folder_path}/ux.npy")
sim_data.uy = np.load(f"{folder_path}/uy.npy")
sim_data.eint = np.load(f"{folder_path}/eint.npy")
sim_data.ps = np.load(f"{folder_path}/ps.npy")

print("Finished loading data.")

# ============================================================================
# TEMPERATURE UNIT
# ============================================================================
#
# Convert Athena pressure/rho units into Kelvin.
# This lets compute_cooling_rate() reproduce exactly the same
# temperatures as the simulation.
#
# T_phys = T_unit * P/rho
#
# ============================================================================

T_unit = float(
    np.median(
        sim_data.temp[0]
        * sim_data.rho[0]
        / sim_data.pressure[0]
    )
)

print(f"T_unit = {T_unit:.3f} K")

# ============================================================================
# COMPUTE TRUE PIXEL PDF
# ============================================================================

print()
print("=" * 70)
print("Computing true PDFs")
print("=" * 70)

temp_pdf = sim_data.calc_pixel_pdf(
    bins=bins
)

temp_pdf /= (
    temp_pdf.sum(
        axis=1,
        keepdims=True,
    )
    + 1e-12
)

nt, nb, nx, ny = temp_pdf.shape

print()

print(f"Snapshots : {nt}")
print(f"Bins       : {nb}")
print(f"Grid       : {nx} x {ny}")

# ============================================================================
# CNN PREDICTION
# ============================================================================

print()
print("=" * 70)
print("Predicting CNN PDFs")
print("=" * 70)

#
# New version
#
# Instead of calling snapshot_pred()
# for every snapshot,
# load the network only ONCE.
#

conv_temp_pdf, gate_maps, vort_maps = batch_predict_with_gate(
    sim_data,
    downsample,
    resolution,
    device,
)

print()

print("CNN prediction complete.")

# ============================================================================
# COARSE GRAINED FIELDS
# ============================================================================

print()
print("=" * 70)
print("Coarse graining simulation fields")
print("=" * 70)

cg_rho = np.zeros(
    (
        nt,
        nx,
        ny,
    )
)

cg_temp = np.zeros_like(cg_rho)

cg_pressure = np.zeros_like(cg_rho)

for t in tqdm(range(nt)):

    cg_rho[t] = sim_data.coarse_grain(
        sim_data.rho[t]
    )

    cg_temp[t] = sim_data.coarse_grain(
        sim_data.temp[t]
    )

    cg_pressure[t] = sim_data.coarse_grain(
        sim_data.pressure[t]
    )

print()

print("Finished coarse graining.")

# ============================================================================
# COOLING CALCULATIONS
# ============================================================================

print()
print("=" * 70)
print("Cooling calculations")
print("=" * 70)

mu = 0.62

true_cool = np.zeros_like(cg_rho)

true_iso_cool = np.zeros_like(cg_rho)

cnn_cool = np.zeros_like(cg_rho)

# ============================================================================
# TRUE FINE COOLING
# ============================================================================

print()

print("Computing true fine cooling...")

for t in tqdm(range(nt)):

    rho = sim_data.rho[t]

    temp = sim_data.temp[t]

    number_density = rho / mu

    lam = lambda_cool(
        temp,
    )

    fine = (
        lam
        * number_density**2
        * 1.975e27
    )

    true_cool[t] = sim_data.coarse_grain(
        fine
    )

# ============================================================================
# TRUE ISOBARIC COOLING
# ============================================================================

print()

print("Computing true isobaric cooling...")

for t in tqdm(range(nt)):

    true_iso_cool[t] = compute_cooling_rate(

        temp_pdf[t],

        temp_centers,

        pressure=cg_pressure[t],

        is_pdf=True,

        is_isobaric=True,

        T_unit=T_unit,

    )

# ============================================================================
# CNN ISOBARIC COOLING
# ============================================================================

print()

print("Computing CNN isobaric cooling...")

for t in tqdm(range(nt)):

    cnn_cool[t] = compute_cooling_rate(

        conv_temp_pdf[t],

        temp_centers,

        pressure=cg_pressure[t],

        is_pdf=True,

        is_isobaric=True,

        T_unit=T_unit,

    )

print()

print("Cooling calculations complete.")

# ============================================================================
# BENCHMARK METRICS
# ============================================================================

print()

print("=" * 70)
print("BENCHMARK METRICS")
print("=" * 70)

print_metrics(

    true_cool.flatten(),

    true_iso_cool.flatten(),

    "Physics Closure Error",

)

print_metrics(

    true_iso_cool.flatten(),

    cnn_cool.flatten(),

    "CNN Prediction Error",

)

print_metrics(

    true_cool.flatten(),

    cnn_cool.flatten(),

    "Total Error",

)

# ============================================================================
# GLOBAL NORMALIZATION
# ============================================================================

positive = np.concatenate(

    [

        true_iso_cool[true_iso_cool > 0],

        cnn_cool[cnn_cool > 0],

    ]

)

if len(positive):

    cool_vmin = max(

        np.percentile(
            positive,
            1,
        ),

        1e-20,

    )

    cool_vmax = np.percentile(

        positive,

        99,

    )

else:

    cool_vmin = 1e-20

    cool_vmax = 1e5

rho_vmin = np.log10(

    np.percentile(

        sim_data.rho,

        1,

    )

)

rho_vmax = np.log10(

    np.percentile(

        sim_data.rho,

        99,

    )

)

print()

print("Finished initialization.")

# ============================================================================
# PDF ANIMATION
# ============================================================================

if RUN_PDF_ANIMATION:

    fig = plt.figure(figsize=(ny * 2.2, nx * 1.8))

    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[3, 1],
    )

    # ----------------------------------------------------------
    # PDF GRID
    # ----------------------------------------------------------

    pdf_axes = np.empty((nx, ny), dtype=object)

    sub_gs = gs[0].subgridspec(nx, ny)

    for i in range(nx):

        for j in range(ny):

            ax = fig.add_subplot(sub_gs[i, j])

            pdf_axes[i, j] = ax

            for spine in ax.spines.values():

                spine.set_visible(True)

                spine.set_color("grey")

                spine.set_linewidth(0.3)

            ax.set_xticks([])

            ax.set_yticks([])

            ax.set_xlim(0, nb - 1)

            ax.set_yscale("log")

            ax.set_ylim(1e-5, 1.1)

            #
            # Highlight cooling region
            #

            ax.axvspan(
                active_bin_start,
                active_bin_end,
                color="green",
                alpha=0.12,
                lw=0,
            )

    # ----------------------------------------------------------
    # Temperature panel
    # ----------------------------------------------------------

    temp_ax = fig.add_subplot(gs[1])

    temp_im = temp_ax.imshow(

        np.log10(cg_temp[0] + 1e-8),

        origin="lower",

        cmap="inferno",

        norm=norm,

    )

    temp_ax.set_title("Coarse Temperature")

    cbar = plt.colorbar(
        temp_im,
        ax=temp_ax,
        fraction=0.046,
    )

    cbar.set_label(

        r"$\log_{10}(T)$",

        fontsize=18,

    )

    # ----------------------------------------------------------
    # PDF lines
    # ----------------------------------------------------------

    x = np.arange(nb)

    lines = np.empty((nx, ny), dtype=object)

    labels = np.empty((nx, ny), dtype=object)

    for i in range(nx):

        for j in range(ny):

            line, = pdf_axes[i, j].plot([], [], lw=1)

            txt = pdf_axes[i, j].text(

                0.97,

                0.97,

                "",

                transform=pdf_axes[i, j].transAxes,

                fontsize=5,

                ha="right",

                va="top",

            )

            lines[i, j] = line

            labels[i, j] = txt

    # ----------------------------------------------------------
    # INIT
    # ----------------------------------------------------------

    def init():

        for i in range(nx):

            for j in range(ny):

                lines[i, j].set_data([], [])

                labels[i, j].set_text("")

                pdf_axes[i, j].set_facecolor("black")

        return [temp_im]

    # ----------------------------------------------------------
    # UPDATE
    # ----------------------------------------------------------

    def update(frame):

        pdf = temp_pdf[frame]

        for i in range(nx):

            for j in range(ny):

                ii = nx - 1 - i

                y = pdf[:, ii, j]

                exp_val = np.sum(

                    y *

                    log_temp_centers

                )

                bg = cmap(

                    norm(

                        exp_val

                    )

                )

                pdf_axes[i, j].set_facecolor(bg)

                lum = (

                    0.299 * bg[0]

                    + 0.587 * bg[1]

                    + 0.114 * bg[2]

                )

                colour = (

                    "white"

                    if lum < 0.5

                    else "black"

                )

                lines[i, j].set_color(colour)

                labels[i, j].set_color(colour)

                lines[i, j].set_data(

                    x,

                    y + 1e-8,

                )

                labels[i, j].set_text(

                    f"{cg_temp[frame,ii,j]:.1e}"

                )

        temp_im.set_data(

            np.log10(

                cg_temp[frame]

                + 1e-8

            )

        )

        fig.suptitle(

            f"Temperature PDF  |  t={frame}",

            fontsize=32,

        )

        if frame == 0:

            fig.savefig(

                first_frame_path,

                dpi=300,

            )

            plt.imsave(

                temp_frame_path,

                np.log10(

                    cg_temp[0]

                    + 1e-8

                ),

                cmap="inferno",

            )

        return [temp_im]

    # ----------------------------------------------------------
    # Animation
    # ----------------------------------------------------------

    print()

    print("Creating PDF animation...")

    anim = animation.FuncAnimation(

        fig,

        update,

        init_func=init,

        frames=nt,

        blit=False,

    )

    anim.save(

        gif_path,

        writer="pillow",

        fps=10,

    )

    plt.close(fig)

    print("Finished PDF animation.")

# ============================================================================
# TRUE vs CNN PDF COMPARISON
# ============================================================================

if RUN_PDF_COMPARE_ANIMATION:

    print()
    print("=" * 70)
    print("Creating PDF comparison animation")
    print("=" * 70)

    snapshot_compare_path = os.path.join(
        PDF_MOCKS_DIR,
        "pdf_compare_t0.png",
    )

    fig2 = plt.figure(figsize=(ny * 4.0, nx * 1.8))

    gs2 = fig2.add_gridspec(
        1,
        3,
        width_ratios=[1, 1, 0.05],
        top=0.90,
        wspace=0.15,
    )

    fig2.text(
        0.24,
        0.92,
        "TRUE PDFs",
        fontsize=32,
        ha="center",
        weight="bold",
    )

    fig2.text(
        0.72,
        0.92,
        "CNN PDFs",
        fontsize=32,
        ha="center",
        weight="bold",
    )

    # ---------------------------------------------------------
    # TRUE GRID
    # ---------------------------------------------------------

    true_axes = np.empty((nx, ny), dtype=object)

    gs_left = gs2[0].subgridspec(nx, ny)

    for i in range(nx):

        for j in range(ny):

            ax = fig2.add_subplot(gs_left[i, j])

            true_axes[i, j] = ax

            ax.set_xticks([])

            ax.set_yticks([])

            ax.set_xlim(0, nb - 1)

            ax.set_yscale("log")

            ax.set_ylim(1e-5, 1.1)

            for s in ax.spines.values():

                s.set_color("grey")

                s.set_linewidth(0.3)

            ax.axvspan(

                active_bin_start,

                active_bin_end,

                color="green",

                alpha=0.12,

                lw=0,

            )

    # ---------------------------------------------------------
    # CNN GRID
    # ---------------------------------------------------------

    pred_axes = np.empty((nx, ny), dtype=object)

    gs_right = gs2[1].subgridspec(nx, ny)

    for i in range(nx):

        for j in range(ny):

            ax = fig2.add_subplot(gs_right[i, j])

            pred_axes[i, j] = ax

            ax.set_xticks([])

            ax.set_yticks([])

            ax.set_xlim(0, nb - 1)

            ax.set_yscale("log")

            ax.set_ylim(1e-5, 1.1)

            for s in ax.spines.values():

                s.set_color("grey")

                s.set_linewidth(0.3)

            ax.axvspan(

                active_bin_start,

                active_bin_end,

                color="green",

                alpha=0.12,

                lw=0,

            )

    # ---------------------------------------------------------
    # Shared temperature colour bar
    # ---------------------------------------------------------

    cbar_ax = fig2.add_subplot(gs2[2])

    sm = plt.cm.ScalarMappable(

        cmap=cmap,

        norm=norm,

    )

    sm.set_array([])

    cb = plt.colorbar(

        sm,

        cax=cbar_ax,

    )

    cb.set_label(

        r"$\langle \log_{10}(T)\rangle$",

        fontsize=18,

    )

    # ---------------------------------------------------------
    # Lines & labels
    # ---------------------------------------------------------

    x = np.arange(nb)

    true_lines = np.empty((nx, ny), dtype=object)

    pred_lines = np.empty((nx, ny), dtype=object)

    true_text = np.empty((nx, ny), dtype=object)

    pred_text = np.empty((nx, ny), dtype=object)

    for i in range(nx):

        for j in range(ny):

            lt, = true_axes[i, j].plot([], [], lw=1)

            lp, = pred_axes[i, j].plot([], [], lw=1)

            tt = true_axes[i, j].text(

                0.97,

                0.97,

                "",

                fontsize=5,

                ha="right",

                va="top",

                transform=true_axes[i, j].transAxes,

            )

            pt = pred_axes[i, j].text(

                0.97,

                0.97,

                "",

                fontsize=5,

                ha="right",

                va="top",

                transform=pred_axes[i, j].transAxes,

            )

            true_lines[i, j] = lt

            pred_lines[i, j] = lp

            true_text[i, j] = tt

            pred_text[i, j] = pt

    # ---------------------------------------------------------
    # UPDATE
    # ---------------------------------------------------------

    def update_compare(frame):

        true_pdf = temp_pdf[frame]

        pred_pdf = conv_temp_pdf[frame]

        for i in range(nx):

            for j in range(ny):

                ii = nx - 1 - i

                yt = true_pdf[:, ii, j]

                yp = pred_pdf[:, ii, j]

                exp_true = np.sum(

                    yt *

                    log_temp_centers

                )

                exp_pred = np.sum(

                    yp *

                    log_temp_centers

                )

                bg_true = cmap(

                    norm(

                        exp_true

                    )

                )

                bg_pred = cmap(

                    norm(

                        exp_pred

                    )

                )

                true_axes[i, j].set_facecolor(bg_true)

                pred_axes[i, j].set_facecolor(bg_pred)

                lum = (

                    0.299 * bg_true[0]

                    + 0.587 * bg_true[1]

                    + 0.114 * bg_true[2]

                )

                tc = "white" if lum < 0.5 else "black"

                lum = (

                    0.299 * bg_pred[0]

                    + 0.587 * bg_pred[1]

                    + 0.114 * bg_pred[2]

                )

                pc = "white" if lum < 0.5 else "black"

                true_lines[i, j].set_color(tc)

                pred_lines[i, j].set_color(pc)

                true_text[i, j].set_color(tc)

                pred_text[i, j].set_color(pc)

                true_lines[i, j].set_data(

                    x,

                    yt + 1e-8,

                )

                pred_lines[i, j].set_data(

                    x,

                    yp + 1e-8,

                )

                #
                # Display both true cooling values
                #

                tf = true_cool[frame, ii, j]

                ti = true_iso_cool[frame, ii, j]

                cp = cnn_cool[frame, ii, j]

                true_text[i, j].set_text(

                    f"F:{tf:.1e}\nI:{ti:.1e}"

                )

                pred_text[i, j].set_text(

                    f"{cp:.1e}"

                )

        fig2.suptitle(

            f"True vs CNN PDFs   t={frame}",

            fontsize=36,

        )

        if frame == 0:

            fig2.savefig(

                snapshot_compare_path,

                dpi=300,

            )

    # ---------------------------------------------------------

    anim2 = animation.FuncAnimation(

        fig2,

        update_compare,

        frames=nt,

        blit=False,

    )

    anim2.save(

        gif_compare,

        writer="pillow",

        fps=10,

    )

    plt.close(fig2)

    print("Finished PDF comparison.")

# ============================================================================
# GLOBAL COOLING DIAGNOSTICS
# ============================================================================

if RUN_COOLING_SCATTER:

    print()
    print("=" * 70)
    print("Creating cooling diagnostics")
    print("=" * 70)

    from matplotlib.colors import LogNorm

    SCATTER_MIN = 1.0

    flat_temp = cg_temp.flatten()

    flat_true = true_cool.flatten()

    flat_iso = true_iso_cool.flatten()

    flat_cnn = cnn_cool.flatten()

    # ------------------------------------------------------------
    # helper
    # ------------------------------------------------------------

    def running_median(ax, x, y, bins=25):

        logx = np.log10(x)

        edges = np.linspace(
            logx.min(),
            logx.max(),
            bins + 1,
        )

        idx = np.digitize(
            logx,
            edges,
        )

        xs = []

        med = []

        lo = []

        hi = []

        for b in range(1, bins + 1):

            mask = idx == b

            if mask.sum() < 10:
                continue

            xs.append(
                10 ** (
                    0.5 *
                    (
                        edges[b - 1] +
                        edges[b]
                    )
                )
            )

            vals = y[mask]

            med.append(
                np.median(vals)
            )

            lo.append(
                np.percentile(vals,16)
            )

            hi.append(
                np.percentile(vals,84)
            )

        if len(xs):

            ax.plot(
                xs,
                med,
                color="cyan",
                lw=2,
                label="Median",
            )

            ax.fill_between(
                xs,
                lo,
                hi,
                color="cyan",
                alpha=0.2,
            )

            ax.legend(
                fontsize=8,
            )

    # ------------------------------------------------------------
    # figure
    # ------------------------------------------------------------

    fig_sc, axs = plt.subplots(

        2,

        2,

        figsize=(14,12),

    )

    ax1 = axs[0,0]

    ax2 = axs[0,1]

    ax3 = axs[1,0]

    ax4 = axs[1,1]

    # ============================================================
    # 1
    # Fine vs Isobaric
    # ============================================================

    mask = (

        (flat_true > SCATTER_MIN)

        &

        (flat_iso > SCATTER_MIN)

    )

    x = flat_true[mask]

    y = flat_iso[mask]

    t = flat_temp[mask]

    hb = ax1.hexbin(

        x,

        y,

        C=t,

        reduce_C_function=np.median,

        gridsize=60,

        cmap="plasma",

        norm=LogNorm(

            vmin=1e3,

            vmax=1e8,

        ),

        xscale="log",

        yscale="log",

        mincnt=1,

    )

    lim = [

        min(x.min(),y.min()),

        max(x.max(),y.max())

    ]

    ax1.plot(

        lim,

        lim,

        "r--",

    )

    ax1.set_xscale("log")

    ax1.set_yscale("log")

    ax1.set_title("True Fine vs True Isobaric")

    ax1.set_xlabel("True Fine")

    ax1.set_ylabel("True Isobaric")

    running_median(ax1,x,y)

    plt.colorbar(

        hb,

        ax=ax1,

        label="Median Temperature",

    )

    # ============================================================
    # 2
    # Isobaric vs CNN
    # ============================================================

    mask = (

        (flat_iso > SCATTER_MIN)

        &

        (flat_cnn > SCATTER_MIN)

    )

    x = flat_iso[mask]

    y = flat_cnn[mask]

    hb = ax2.hexbin(

        x,

        y,

        gridsize=60,

        bins="log",

        cmap="viridis",

        xscale="log",

        yscale="log",

        mincnt=1,

    )

    lim = [

        min(x.min(),y.min()),

        max(x.max(),y.max())

    ]

    ax2.plot(

        lim,

        lim,

        "r--",

    )

    ax2.set_xscale("log")

    ax2.set_yscale("log")

    ax2.set_title("True Isobaric vs CNN")

    ax2.set_xlabel("True Isobaric")

    ax2.set_ylabel("CNN")

    running_median(ax2,x,y)

    plt.colorbar(

        hb,

        ax=ax2,

        label="Counts",

    )

    # ============================================================
    # 3
    # Fine vs CNN
    # ============================================================

    mask = (

        (flat_true > SCATTER_MIN)

        &

        (flat_cnn > SCATTER_MIN)

    )

    x = flat_true[mask]

    y = flat_cnn[mask]

    hb = ax3.hexbin(

        x,

        y,

        gridsize=60,

        bins="log",

        cmap="viridis",

        xscale="log",

        yscale="log",

        mincnt=1,

    )

    lim = [

        min(x.min(),y.min()),

        max(x.max(),y.max())

    ]

    ax3.plot(

        lim,

        lim,

        "r--",

    )

    ax3.set_xscale("log")

    ax3.set_yscale("log")

    ax3.set_title("True Fine vs CNN")

    ax3.set_xlabel("True Fine")

    ax3.set_ylabel("CNN")

    running_median(ax3,x,y)

    plt.colorbar(

        hb,

        ax=ax3,

        label="Counts",

    )

    # ============================================================
    # residual
    # ============================================================

    mask = (

        (flat_iso > SCATTER_MIN)

        &

        (flat_cnn > SCATTER_MIN)

    )

    x = flat_iso[mask]

    resid = np.log10(

        flat_cnn[mask]

        /

        flat_iso[mask]

    )

    hb = ax4.hexbin(

        x,

        resid,

        gridsize=60,

        bins="log",

        cmap="viridis",

        xscale="log",

        mincnt=1,

    )

    ax4.axhline(

        0,

        color="r",

        linestyle="--",

    )

    ax4.set_xscale("log")

    ax4.set_xlabel("True Isobaric")

    ax4.set_ylabel(r"$\log_{10}$(CNN / True)")

    ax4.set_title("CNN Residuals")

    plt.colorbar(

        hb,

        ax=ax4,

        label="Counts",

    )

    fig_sc.tight_layout()

    fig_sc.savefig(

        os.path.join(

            PDF_MOCKS_DIR,

            "pdf_cooling_scatter_threeway.png",

        ),

        dpi=250,

    )

    plt.close(fig_sc)

# ============================================================================
# COOLING HISTOGRAMS
# ============================================================================

if RUN_COOLING_HISTOGRAM:

    print()
    print("=" * 70)
    print("Cooling histograms")
    print("=" * 70)

    fields = {

        "True Fine": true_cool.flatten(),

        "True Isobaric": true_iso_cool.flatten(),

        "CNN": cnn_cool.flatten(),

    }

    colours = [

        "steelblue",

        "darkorange",

        "forestgreen",

    ]

    fig, (ax1, ax2) = plt.subplots(

        1,

        2,

        figsize=(14,5),

    )

    # ----------------------------------------------------------
    # Zero fraction
    # ----------------------------------------------------------

    zero_fraction = [

        np.mean(v == 0.0) * 100

        for v in fields.values()

    ]

    bars = ax1.bar(

        list(fields.keys()),

        zero_fraction,

        color=colours,

    )

    for b, f in zip(bars, zero_fraction):

        ax1.text(

            b.get_x() + b.get_width()/2,

            b.get_height()+0.5,

            f"{f:.1f}%",

            ha="center",

        )

    ax1.set_ylabel("% zero cooling")

    ax1.set_title("Zero Cooling Fraction")

    # ----------------------------------------------------------
    # Positive cooling distribution
    # ----------------------------------------------------------

    for (label, values), c in zip(fields.items(), colours):

        values = values[values > 0]

        if len(values) == 0:

            continue

        ax2.hist(

            np.log10(values),

            bins=80,

            density=True,

            histtype="step",

            linewidth=2,

            label=label,

            color=c,

        )

    ax2.set_xlabel(r"$\log_{10}(\Lambda)$")

    ax2.set_ylabel("Probability Density")

    ax2.set_yscale("log")

    ax2.legend()

    ax2.set_title("Cooling Distribution")

    fig.tight_layout()

    fig.savefig(

        os.path.join(

            PDF_MOCKS_DIR,

            "pdf_cooling_histogram.png",

        ),

        dpi=250,

    )

    plt.close(fig)

# ============================================================================
# TRUE FINE vs TRUE ISOBARIC vs CNN COOLING ANIMATION
# ============================================================================

if RUN_COOLING_COMPARE_ANIMATION:

    print()
    print("=" * 70)
    print("Cooling comparison animation")
    print("=" * 70)

    cooling_path = os.path.join(
        PDF_MOCKS_DIR,
        "pdf_cooling_compare.gif",
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
    )

    norm_cool = colors.LogNorm(
        vmin=3e1,
        vmax=2e3,
    )

    im_fine = axes[0].imshow(
        np.clip(
            true_cool[0],
            cool_vmin,
            None,
        ),
        origin="lower",
        cmap="viridis",
        norm=norm_cool,
    )

    im_iso = axes[1].imshow(
        np.clip(
            true_iso_cool[0],
            cool_vmin,
            None,
        ),
        origin="lower",
        cmap="viridis",
        norm=norm_cool,
    )

    im_cnn = axes[2].imshow(
        np.clip(
            cnn_cool[0],
            cool_vmin,
            None,
        ),
        origin="lower",
        cmap="viridis",
        norm=norm_cool,
    )

    axes[0].set_title("True Fine")
    axes[1].set_title("True Isobaric")
    axes[2].set_title("CNN")

    for ax in axes:
        ax.set_xlabel("Y")
        ax.set_ylabel("X")

    cb = fig.colorbar(
        im_fine,
        ax=axes,
        fraction=0.035,
        pad=0.02,
    )
    cb.set_label("Cooling Rate")

    def update(frame):

        im_fine.set_data(
            np.clip(
                true_cool[frame],
                cool_vmin,
                None,
            )
        )

        im_iso.set_data(
            np.clip(
                true_iso_cool[frame],
                cool_vmin,
                None,
            )
        )

        im_cnn.set_data(
            np.clip(
                cnn_cool[frame],
                cool_vmin,
                None,
            )
        )

        fig.suptitle(
            f"Cooling Comparison   t={frame}",
            fontsize=20,
        )

        return (
            im_fine,
            im_iso,
            im_cnn,
        )

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=nt,
        blit=False,
    )

    anim.save(
        cooling_path,
        writer="pillow",
        fps=10,
    )

    plt.close(fig)

    print("=" * 70)
    print("Finished cooling comparison animation.")
    print("=" * 70)

# ============================================================================
# DENSITY / GATE ANIMATION
# ============================================================================

if RUN_DENSITY_GATE_ANIMATION:

    print()
    print("=" * 70)
    print("Density / Gate animation")
    print("=" * 70)

    gate_path = os.path.join(
        PDF_MOCKS_DIR,
        "density_gate.gif",
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14,6),
    )

    # ---------------------------------------------------------
    # Fine density
    # ---------------------------------------------------------

    im_fine = axes[0].imshow(
        np.log10(sim_data.rho[0] + 1e-12),
        origin="lower",
        cmap="viridis",
        vmin=rho_vmin,
        vmax=rho_vmax,
    )

    axes[0].set_title("Fine Density")

    fig.colorbar(
        im_fine,
        ax=axes[0],
        fraction=0.046,
    )

    # ---------------------------------------------------------
    # Coarse density
    # ---------------------------------------------------------

    im_cg = axes[1].imshow(
        np.log10(cg_rho[0] + 1e-12),
        origin="lower",
        cmap="viridis",
        vmin=rho_vmin,
        vmax=rho_vmax,
    )

    axes[1].set_title("Coarse Density")

    fig.colorbar(
        im_cg,
        ax=axes[1],
        fraction=0.046,
    )

    # ---------------------------------------------------------
    # CNN Gate
    # ---------------------------------------------------------

    im_gate = axes[2].imshow(
        gate_maps[0],
        origin="lower",
        cmap="inferno",
        vmin=0,
        vmax=1,
    )

    axes[2].set_title("CNN Gate")

    fig.colorbar(
        im_gate,
        ax=axes[2],
        fraction=0.046,
    )

    for ax in axes:

        ax.set_xlabel("Y")

        ax.set_ylabel("X")

    def update(frame):

        im_fine.set_data(
            np.log10(
                sim_data.rho[frame] + 1e-12
            )
        )

        im_cg.set_data(
            np.log10(
                cg_rho[frame] + 1e-12
            )
        )

        im_gate.set_data(
            gate_maps[frame]
        )

        fig.suptitle(
            f"Density / Gate   t={frame}",
            fontsize=18,
        )

        return (
            im_fine,
            im_cg,
            im_gate,
        )

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=nt,
        blit=False,
    )

    anim.save(
        gate_path,
        writer="pillow",
        fps=10,
    )

    plt.close(fig)

    print("=" * 70)
    print("Finished density / gate animation.")
    print("=" * 70)

