import numpy as np
import struct
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

file_path = "mocks/KH.hydro_w.00000.bin"
header_offset = 4257
variable_size = 4
shape = (512, 512)

with open(file_path, 'rb') as f:
    f.seek(header_offset)
    raw_data = np.fromfile(f, dtype=np.float32)
    
    expected_size = np.prod(shape) * 6
    if raw_data.size != expected_size:
        raise ValueError(f"Data size mismatch: expected {expected_size} elements, but found {raw_data.size}")
    
    high_res_rho = raw_data[0:shape[0]*shape[1]].reshape(shape)
    high_res_velx = raw_data[shape[0]*shape[1]:2*shape[0]*shape[1]].reshape(shape)
    high_res_vely = raw_data[2*shape[0]*shape[1]:3*shape[0]*shape[1]].reshape(shape)
    high_res_velz = raw_data[3*shape[0]*shape[1]:4*shape[0]*shape[1]].reshape(shape)
    high_res_eint = raw_data[4*shape[0]*shape[1]:5*shape[0]*shape[1]].reshape(shape)
    high_res_s_00 = raw_data[5*shape[0]*shape[1]:6*shape[0]*shape[1]].reshape(shape)

plt.imshow(high_res_rho, cmap='inferno', norm=LogNorm())
plt.colorbar(label="Density")
plt.title("Density (dens)")
plt.savefig("mocks/bin_sim_density.jpg", dpi=500)
