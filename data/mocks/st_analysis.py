import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import data_preprocess
from data_preprocess import simulation_data
from feedforward_nn.fnn import snapshot_pred, feedforwardNN
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from tqdm import tqdm
from scipy.signal import correlate2d

resolution = (256, 256)  # Resolution of the simulation
downsample = 8
file_path = f"/data3/home/dipayandatta/Subgrid_CGM_Models/data/files/Subgrid CGM Models/Without_cooling/rk2, plm/{resolution[0]}_{resolution[1]}_prateek/bin"
save_path = f"mocks/src/{resolution}_{downsample}/"
os.makedirs(save_path, exist_ok=True)

sim_data = simulation_data()
sim_data.resolution = resolution
sim_data.down_sample = downsample
sim_data.rho = np.load(f"../feedforward_nn/data_saves/{resolution}_{downsample}/rho.npy")
sim_data.temp = np.load(f"../feedforward_nn/data_saves/{resolution}_{downsample}/temp.npy")
sim_data.pressure = np.load(f"../feedforward_nn/data_saves/{resolution}_{downsample}/pressure.npy")
sim_data.ux = np.load(f"../feedforward_nn/data_saves/{resolution}_{downsample}/ux.npy")
sim_data.uy = np.load(f"../feedforward_nn/data_saves/{resolution}_{downsample}/uy.npy")
sim_data.eint = np.load(f"../feedforward_nn/data_saves/{resolution}_{downsample}/eint.npy")
sim_data.ps = np.load(f"../feedforward_nn/data_saves/{resolution}_{downsample}/ps.npy")
print("Input data loaded")

high_res_rho = sim_data.rho
cg_rho = np.zeros((high_res_rho.shape[0], high_res_rho.shape[1] // sim_data.down_sample, high_res_rho.shape[2] // sim_data.down_sample))
fmcl_data = np.zeros((sim_data.rho.shape[0], sim_data.rho.shape[1] // sim_data.down_sample, sim_data.rho.shape[2] // sim_data.down_sample))
for i in tqdm(range(high_res_rho.shape[0]), desc = "Calculating cold gas mass fraction"):
    fmcl_data[i] = sim_data.calc_fmcl(sim_data.rho[i], sim_data.temp[i])

source_term = sim_data.calc_source_term()   
source_term_pred = np.zeros_like(source_term)

for i in tqdm(range(source_term.shape[0]), desc = "Predicting source term"):
    source_term_pred[i] = snapshot_pred(sim_data.rho[i], sim_data.temp[i], sim_data.pressure[i], \
                                        sim_data.ux[i], sim_data.uy[i], sim_data.eint[i], \
                                        downsample, (sim_data.resolution[0], sim_data.resolution[1]))
residuals = source_term - source_term_pred

# Perform the FFT of source terms along x
st_x = np.mean(source_term, axis=1)
st_x_pred = np.mean(source_term_pred, axis=1)

fst_x = np.fft.fft(st_x, axis=1)
fst_x_pred = np.fft.fft(st_x_pred, axis=1)
k_val = np.fft.fftfreq(st_x.shape[1], d=10/cg_rho.shape[1])

fst_x = np.fft.fftshift(fst_x, axes=1)
fst_x_pred = np.fft.fftshift(fst_x_pred, axes=1)
k_val = np.fft.fftshift(k_val)

t_idxs = 7
colors = plt.cm.plasma(np.linspace(1, 0, t_idxs))
plt.figure(figsize=(10, 6))
for t_idx in tqdm(np.linspace(0, fst_x.shape[0] - 1, t_idxs, dtype=int), desc="Plotting FFT along X"):
    plt.plot(k_val, np.abs(fst_x[t_idx]), color=colors[int((t_idx/fst_x.shape[0])*t_idxs)], label=f'{t_idx/fst_x.shape[0]:.2} Myr')
plt.xlabel(r'$k\,(pc^{-1})$')
plt.xlim(0.0, np.max(k_val))
plt.ylabel(r'$C_k$')
plt.legend()
plt.title(rf'Source Term FFT along X for ${resolution[0]} \times {resolution[1]}$ resolution')
plt.savefig(f"{save_path}/fft_x.png", dpi=300)
plt.clf()

def compute_structure_function_2d(field):
    field = field - np.mean(field)

    f2 = field**2
    autocorr = correlate2d(field, field, mode='full')
    norm = correlate2d(np.ones_like(field), np.ones_like(field), mode='full')

    sq_diff = correlate2d(f2, np.ones_like(field), mode='full') \
            + correlate2d(np.ones_like(field), f2, mode='full') \
            - 2 * autocorr
    structure = sq_diff / norm 

    return structure

def radial_average(structure_2d):
    center = np.array(structure_2d.shape) // 2
    y, x = np.indices(structure_2d.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int32)

    tbin = np.bincount(r.ravel(), structure_2d.ravel())
    nr = np.bincount(r.ravel())
    radial_prof = tbin / nr
    return np.arange(len(radial_prof)), radial_prof

plt.figure(figsize=(10, 6))
for t_idx in tqdm(np.linspace(0, source_term.shape[0] - 1, t_idxs, dtype=int), desc="Plotting Structure Function"):
    structure_2d = compute_structure_function_2d(source_term[t_idx])
    r, radial_prof = radial_average(structure_2d)
    plt.plot(r * 10/cg_rho.shape[1], radial_prof, color=colors[int((t_idx/source_term.shape[0])*t_idxs)], label=f'{t_idx/source_term.shape[0]:.2} Myr')
plt.xlabel(r'$r\,(pc)$')
plt.xlim(0.0, 1.5)
plt.ylabel(r'$S(r)$')
plt.title(rf'Structure Function for ${resolution[0]} \times {resolution[1]}$ resolution')
plt.legend()
plt.savefig(f"{save_path}/structure_function.png", dpi=300)
plt.clf()
