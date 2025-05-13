import numpy as np 
import os
import skimage.measure
import matplotlib.pyplot as plt

class simulation_data():

    T_cutoff: float = 1e5 # K

    def __init__(self: "simulation_data") -> None:

        # time step X length X width arrays
        self.rho: np.ndarray = None
        self.temp: np.ndarray = None
        self.ux: np.ndarray = None
        self.uy: np.ndarray = None
        self.uz: np.ndarray = None
        self.eint: np.ndarray = None
        self.ps: np.ndarray = None

    def input_data(self: "simulation_data", filepath: str) -> None:
        
        os.chdir(filepath)
        # self.rho = np.loadtxt(filepath/"rho.txt")
        # self.temp = np.loadtxt(filepath/"temp.txt")     
        # self.u = np.loadtxt(filepath/"u.txt")
        pass

    def coarse_grain(self: "simulation_data", quan: np.ndarray, down_sample: int) -> np.ndarray:
        return skimage.measure.block_reduce(quan, (down_sample, down_sample), np.mean)
    
    def calc_fmcl(self: "simulation_data", rho: np.ndarray, temp: np.ndarray, down_sample: int) -> np.ndarray:
        rho_block = rho.reshape(rho.shape[0] // down_sample, down_sample, rho.shape[1] // down_sample, down_sample)
        temp_block = temp.reshape(temp.shape[0] // down_sample, down_sample, temp.shape[1] // down_sample, down_sample)
        fc = np.sum(rho_block * (temp_block < self.T_cutoff), axis=(1, 3))
        fh = np.sum(rho_block * (temp_block > self.T_cutoff), axis=(1, 3))
        fmcl = fc / (fc + fh)
        return fmcl

    def calc_deriv(self: "simulation_data", quan: np.ndarray) -> np.ndarray:
        pass
    