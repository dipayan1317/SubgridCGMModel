import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

debug_rho = np.load('debug_rho.npy')
debug_temp = np.load('debug_temp.npy')
debug_ux = np.load('debug_ux.npy')
debug_uy = np.load('debug_uy.npy')
debug_ps = np.load('debug_ps.npy')
debug_fmcl = np.load('debug_fmcl.npy')
debug_source = np.load('debug_source_term.npy')

print(np.mean(debug_source[3]), np.std(debug_source[3]))

rho_min, rho_max = np.min(debug_rho), np.max(debug_rho)
temp_min, temp_max = np.min(debug_temp), np.max(debug_temp)

fig, axs = plt.subplots(2, 3, figsize=(15, 5))

axs = axs.flatten()

im0 = axs[0].imshow(debug_rho, cmap='viridis', norm=LogNorm(vmin=rho_min, vmax=rho_max), origin='lower')
# im0 = axs[0].imshow(debug_rho, cmap='viridis', origin='lower')
axs[0].set_title('debug_rho')
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(debug_temp, cmap='plasma', norm=LogNorm(vmin=temp_min, vmax=temp_max), origin='lower')
# im1 = axs[1].imshow(debug_temp, cmap='plasma', origin='lower')
axs[1].set_title('debug_temp')
plt.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(debug_ux, cmap='coolwarm', origin='lower')
axs[2].set_title('debug_ux')
plt.colorbar(im2, ax=axs[2])

im3 = axs[3].imshow(debug_uy, cmap='coolwarm', origin='lower')
axs[3].set_title('debug_uy')
plt.colorbar(im3, ax=axs[3])

im4 = axs[4].imshow(debug_ps, cmap='viridis', origin='lower')
axs[4].set_title('debug_ps')
plt.colorbar(im4, ax=axs[4])

im5 = axs[5].imshow(debug_fmcl, cmap='viridis', origin='lower')
axs[5].set_title('debug_fmcl')
plt.colorbar(im5, ax=axs[5])

plt.tight_layout()
plt.savefig('debug_inp.png')
plt.close()

fig, axes = plt.subplots(1, 5, figsize=(18, 4))  

for i in range(5):
    im = axes[i].imshow(debug_source[i], origin='lower', cmap='viridis')
    axes[i].set_title(f'debug_source[{i}]')
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('debug_out.png')
plt.close()