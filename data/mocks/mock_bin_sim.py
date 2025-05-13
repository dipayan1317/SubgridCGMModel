import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

file_path = "mocks/KH.hydro_w.00000.bin"

header_offset = 4257
variables = ['dens', 'velx', 'vely', 'velz', 'eint', 's_00']
num_vars = len(variables)
dtype = np.float32
shape = (512, 512)  

num_elements_per_var = shape[0] * shape[1]
expected_total_elements = num_elements_per_var * num_vars

with open(file_path, 'rb') as f:
    f.seek(header_offset)
    data = np.fromfile(f, dtype=dtype)

if data.size != expected_total_elements:
    raise ValueError(f"Data size mismatch: expected {expected_total_elements}, got {data.size}")

data_arrays = {}
for i, var in enumerate(variables):
    start = i * num_elements_per_var
    end = (i + 1) * num_elements_per_var
    data_arrays[var] = data[start:end].reshape(shape)

plt.imshow(data_arrays['dens'], cmap='inferno', norm=LogNorm())
plt.colorbar(label="Density")
plt.title("Density (dens)")
plt.savefig("mocks/bin_sim_density.jpg", dpi=500)
plt.clf()
