import h5py
import numpy as np


def load_data(filepath, sample_idx=0):
    with h5py.File(filepath, 'r') as f:
        i1 = f['inputs'][sample_idx, :, :, 0]
        i2 = f['targets'][sample_idx, :, :, 0]

    Ny, Nx = i1.shape
    return i1, i2, Nx, Ny


def load_data_mod(filepath, sample_idx=0):
    with h5py.File(filepath, 'r') as f:
        i1 = f['inputs'][sample_idx, :, :, 0]
        i2 = f['targets'][sample_idx, :, :, 0]
        ph0 = f['phase0'][sample_idx, :, :, 0]

    Nx = i1.shape
    Ny = i1.shape
    return i1, i2, ph0, Nx, Ny


def load_data_kulki(filepath, sample_idx=0):
    with h5py.File(filepath, 'r') as f:
        i1 = f['i_ccd1'][()]
        i2 = f['i_ccd2'][()]
        ph0 = f['ph_obj'][()]
        z1 = f['z_sample_ccd1'][()]
        z2 = f['z_sample_ccd2'][()]

    i1 = i1.astype(np.float32)
    i2 = i2.astype(np.float32)

    Ny, Nx = i1.shape
    return i1, i2, ph0, Nx, Ny, z1, z2