import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())
import data_preprocess
from data_preprocess import simulation_data
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from tqdm import tqdm

file_path = "/data3/home/dipayandatta/Subgrid_CGM_Models/data/files/Subgrid CGM Models/Without_cooling/rk2, plm/1024_1024_prateek/bin"

sim_data = simulation_data()
sim_data.input_data(file_path)
print("Input data loaded")

high_res_rho = sim_data.rho
cg_rho = np.zeros((high_res_rho.shape[0], high_res_rho.shape[1] // sim_data.down_sample, high_res_rho.shape[2] // sim_data.down_sample))
fmcl_data = np.zeros((sim_data.rho.shape[0], sim_data.rho.shape[1] // sim_data.down_sample, sim_data.rho.shape[2] // sim_data.down_sample))
for i in tqdm(range(high_res_rho.shape[0]), desc = "Coarse Graining"):
    cg_rho[i] = sim_data.coarse_grain(high_res_rho[i])
    fmcl_data[i] = sim_data.calc_fmcl(sim_data.rho[i], sim_data.temp[i])
source_term = sim_data.calc_source_term()
    
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

im1 = axs[0, 0].imshow(high_res_rho[0], origin='lower', cmap='plasma', norm=LogNorm())
axs[0, 0].set_title(r'HR ($128 \times 128$) Density')
plt.colorbar(im1, ax=axs[0, 0], fraction=0.046, pad=0.04)

im2 = axs[0, 1].imshow(cg_rho[0], origin='lower', cmap='plasma', norm=LogNorm())
axs[0, 1].set_title(r'CG ($16 \times 16$) Density')
plt.colorbar(im2, ax=axs[0, 1], fraction=0.046, pad=0.04)

im3 = axs[1, 0].imshow(source_term[0], origin='lower', cmap='viridis', vmin = -1, vmax = 1)
axs[1, 0].set_title('Source Term')
plt.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)

im4 = axs[1, 1].imshow(fmcl_data[0], origin='lower', cmap='viridis')
axs[1, 1].set_title(r'$f_{m}^{cl}$')
plt.colorbar(im4, ax=axs[1, 1], fraction=0.046, pad=0.04)

def update_all(frame):
    im1.set_data(high_res_rho[frame])
    im2.set_data(cg_rho[frame])
    im3.set_data(source_term[frame])
    im4.set_data(fmcl_data[frame])
    for ax in axs.flat:
        ax.set_xlabel(f'Timestep: {frame}')
    return [im1, im2, im3, im4]

ani = animation.FuncAnimation(fig, update_all, frames=high_res_rho.shape[0], interval=100, blit=True)
ani.save("mocks/wrc_evolution.mp4", writer='ffmpeg')
plt.close()

hist_data = sim_data.fmcl_hist()
plt.hist(hist_data, bins=10, histtype='step', color='black')
plt.xlabel(r'$f_{m}^{cl}$')
plt.ylabel('Counts')
plt.yscale('log')
plt.title(r"Histogram of Cold gas Mass fraction")
plt.savefig("mocks/wrc_histogram.jpg", dpi=500)
plt.close()
