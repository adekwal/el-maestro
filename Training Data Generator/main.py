import os
import time
import h5py
import numpy as np
import imageio.v2 as imageio
from training_data_gen import generate_data


# Define simulation parameters
data_path = r"C:\Users\Administrator\Desktop\adek files\python\dataset\flowers" # define path to directory with images
nx = int(256) # cropped image size
ny = int(256)
px = 2 * nx  # padded image size
py = 2 * ny
dx = 2.4
dy = dx
n0 = 1  # refractive index
lambda_ = 0.561
z1 = 1.644e4
z2 = 2.1e4
delta_z = z2 - z1
phase_range = [-np.pi, np.pi]
amplitude_range = [0, 0.6]
px_trim_value = 3   # amount of pixels being trimmed from edges

# Define user setup
img_count = 20  # choose the number of images you want to process
output_file = f'flowers_{img_count}_{nx}x{ny}_test4'  # define the output filename (no extension)
ph_amp_dataset = False   # set 'True' if amplitude-phase image else 'False'
save_disc_space = True  # compress data file if needed (recommended)

# Checking propagation condition for angular spectrum
print("Checking propagation condition for angular spectrum:")
print(lambda_ * delta_z < min([px, py]) * dx * dx)
print(lambda_ * z2 < min([px, py]) * dx * dx)
print( )

h5_file = f'{output_file}.h5'
with h5py.File(h5_file, 'w') as h5f:
    h5f.create_dataset('inputs', shape=(0, nx, ny, 1), maxshape=(None, nx, ny, 1), dtype='float32')
    h5f.create_dataset('targets', shape=(0, nx, ny, 1), maxshape=(None, nx, ny, 1), dtype='float32')
    h5f.create_dataset('phase0', shape=(0, nx, ny, 1), maxshape=(None, nx, ny, 1), dtype='float32')

img_files = sorted([f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))])
img_files = img_files[:img_count] if not ph_amp_dataset else img_files[:img_count+1] # +1 applied as amp-ph dataset requires n+1 images

start_time = time.time()
for i in range(0, len(img_files) - (1 if ph_amp_dataset else 0)):
    ph = imageio.imread(os.path.join(data_path, img_files[i])).astype(float)
    amp = imageio.imread(os.path.join(data_path, img_files[i + 1])).astype(float) if ph_amp_dataset else None

    i1, i2, ph0 = generate_data(ph, phase_range, amplitude_range, px_trim_value, z1, z2, lambda_, dx, nx, ny, px, py, amp)  # generate data

    with h5py.File(h5_file, 'a') as h5f:
        for dataset_name, data in zip(['inputs', 'targets', 'phase0'], [i1, i2, ph0]):
            dset = h5f[dataset_name]
            dset.resize((dset.shape[0] + 1), axis=0)
            dset[-1] = data[..., np.newaxis]
    print(f"{i+1}|{img_count}")

end_time = time.time()
print(f"\nExecute time: {end_time - start_time:.0f} [s]")

h5f.close()
print(f"Data has been saved as: {h5_file}")

if save_disc_space:
    print("\nPlease wait to compress file...")
    with h5py.File(h5_file, 'r') as f_in:
        with h5py.File(f'{output_file}_compressed.h5', 'w') as f_out:
            for dataset_name in f_in:
                data = f_in[dataset_name][:]
                f_out.create_dataset(dataset_name, data=data, compression='gzip', compression_opts=9, dtype=data.dtype)
    os.remove(h5_file)
    print(f"File '{output_file}.h5' has been compressed")
