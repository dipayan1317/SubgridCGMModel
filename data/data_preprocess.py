import numpy as np 
import os
import skimage.measure
import matplotlib.pyplot as plt
import bin_convert

class simulation_data():

    T_cutoff: float = 1e5 # K
    down_sample: int = 8
    total_time: float = 1 # Myr
    total_length: float = 10 # pc
    total_width: float = 10 # pc
    resolution: tuple = (128, 128) 

    def __init__(self: "simulation_data") -> None:

        # time step X length X width arrays
        self.rho: np.ndarray = None
        self.temp: np.ndarray = None
        self.pressure: np.ndarray = None
        self.ux: np.ndarray = None
        self.uy: np.ndarray = None
        self.eint: np.ndarray = None
        self.ps: np.ndarray = None

    def input_data(self: "simulation_data", filepath: str) -> None:

        cwd = os.getcwd()
        os.chdir(filepath)
        num_snaps = len([f for f in os.listdir(filepath) if f.endswith('.bin')])

        self.rho = np.zeros((num_snaps, self.resolution[0], self.resolution[1]))
        self.temp = np.zeros_like(self.rho)
        self.pressure = np.zeros_like(self.rho)
        self.ux = np.zeros_like(self.rho)
        self.uy = np.zeros_like(self.rho)
        self.eint = np.zeros_like(self.rho)
        self.ps = np.zeros_like(self.rho)

        for i in range(num_snaps):
            file_data = bin_convert.read_binary(f"KH.hydro_w.{i:05d}.bin")
            self.rho[i] = bin_convert.make_2D_array(file_data, "dens")
            self.ux[i] = bin_convert.make_2D_array(file_data, "velx")
            self.uy[i] = bin_convert.make_2D_array(file_data, "vely")
            self.eint[i] = bin_convert.make_2D_array(file_data, "eint")
            self.ps[i] = bin_convert.make_2D_array(file_data, "s_00")

            self.pressure[i] = 2./3. * self.eint[i]
            self.temp[i] = (self.pressure[i] * 1.59916e-14 / self.rho[i]) * (1. / 1.381e-16)

        os.chdir(cwd)
        return

    def coarse_grain(self: "simulation_data", quan: np.ndarray) -> np.ndarray:
        return skimage.measure.block_reduce(quan, (self.down_sample, self.down_sample), np.mean)
    
    def calc_fmcl(self: "simulation_data", rho: np.ndarray, temp: np.ndarray) -> np.ndarray:
        rho_block = rho.reshape(rho.shape[0] // self.down_sample, self.down_sample, rho.shape[1] // self.down_sample, self.down_sample)
        temp_block = temp.reshape(temp.shape[0] // self.down_sample, self.down_sample, temp.shape[1] // self.down_sample, self.down_sample)
        fc = np.sum(rho_block * (temp_block < self.T_cutoff), axis=(1, 3))
        fh = np.sum(rho_block * (temp_block > self.T_cutoff), axis=(1, 3))
        fmcl = fc / (fc + fh)
        return fmcl
    
    def fmcl_hist(self: "simulation_data") -> None:
        fmcl_data = np.zeros((self.rho.shape[0], self.rho.shape[1] // self.down_sample, self.rho.shape[2] // self.down_sample))
        for i in range(self.rho.shape[0]):
            fmcl_data[i] = self.calc_fmcl(self.rho[i], self.temp[i])
        hist_data = fmcl_data.flatten()
        return hist_data
    
    def calc_source_term(self: "simulation_data") -> np.ndarray:

        fmcl = np.zeros((self.rho.shape[0], self.rho.shape[1] // self.down_sample, self.rho.shape[2] // self.down_sample))
        cg_rho = np.zeros_like(fmcl)
        cg_ux = np.zeros_like(fmcl)
        cg_uy = np.zeros_like(fmcl)
        source_term = np.zeros_like(fmcl)

        for i in range(self.rho.shape[0]):

            fmcl[i] = self.calc_fmcl(self.rho[i], self.temp[i])
            cg_rho[i] = self.coarse_grain(self.rho[i])
            cg_ux[i] = self.coarse_grain(self.ux[i])
            cg_uy[i] = self.coarse_grain(self.uy[i])

            term1 = np.zeros((cg_rho[i].shape[0], cg_rho[i].shape[1]))
            term2 = np.zeros_like(term1)

            if i == 0:
                term1 = (fmcl[i+1] * cg_rho[i+1] - fmcl[i] * cg_rho[i]) / (self.total_time / self.rho.shape[0])
            if i == self.rho.shape[0] - 1:
                term1 = (fmcl[i] * cg_rho[i] - fmcl[i-1] * cg_rho[i-1]) / (self.total_time / self.rho.shape[0])
            else:
                term1 = (fmcl[i+1] * cg_rho[i+1] - fmcl[i-1] * cg_rho[i-1]) / (2 * self.total_time / self.rho.shape[0])

            dx = self.total_length / self.rho.shape[1]
            dy = self.total_width / self.rho.shape[2]
            term2 = cg_ux[i] * np.gradient(fmcl[i] * cg_rho[i], dx, dy)[0] + cg_uy[i] * np.gradient(fmcl[i] * cg_rho[i], dx, dy)[1] \
                + cg_rho[i] * fmcl[i] * (np.gradient(cg_ux[i], dx, dy)[0] + np.gradient(cg_uy[i], dx, dy)[1])
            
            source_term[i] = term1 + term2

        return source_term

    

    