import numpy as np
import os
import imageio.v2 as imageio
import h5py
from training_data_gen import generate_data
import time

# Define simulation parameters
data_path = r"C:\Users\Administrator\Desktop\adek files\python\dataset\flowers" # define path to directory with images
nx = int(256) # cropped image size
ny = int(256)
px = 2 * nx  # padded image size
py = 2 * ny
dx = 2.4
dy = dx
n0 = 1 # refractive index
lambda_ = 0.561
delta_z = 8.2222e3
z1 = 3.5578e3 # distance to the 1st plane
z2 = z1 + delta_z
delta_ph_max = np.pi / 2 # not in use

# Define user setup
output_file = 'flowers_4000_256x256_amp_ph'  # define the output filename (no extension)
img_count = 4000  # choose the number of images you want to process
save_as_h5 = True  # highly recommended as the ResNet uses H5 file
save_disc_space = True  # compress data file if needed

# Checking propagation condition for angular spectrum
print(lambda_ * delta_z < min([px, py]) * dx * dx)
print(lambda_ * z2 < min([px, py]) * dx * dx)
print()

if save_as_h5:
    h5_file = f'{output_file}.h5'

    with h5py.File(h5_file, 'w') as h5f:
        h5f.create_dataset('inputs', shape=(0, nx, ny, 1), maxshape=(None, nx, ny, 1), compression=None,
                           dtype='float32')
        h5f.create_dataset('targets', shape=(0, nx, ny, 1), maxshape=(None, nx, ny, 1), compression=None,
                           dtype='float32')
        h5f.create_dataset('phase0', shape=(0, nx, ny, 1), maxshape=(None, nx, ny, 1), compression=None,
                           dtype='float32')


# img_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
# img_files = img_files[:img_count]
#
# start_time = time.time()
#
# for img_no, img_file in enumerate(img_files, start=1):
#     img = imageio.imread(os.path.join(data_path, img_file)).astype(float)
#     i1, i2, ph0 = generate_data(img, delta_ph_max, z1, z2, lambda_, dx, nx, ny, px, py)  # generate data

img_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
img_files = img_files[:img_count+1] # +1 applied as amp-ph dataset requires n+1 images

start_time = time.time()

for i in range(0, len(img_files) - 1):
    ph = imageio.imread(os.path.join(data_path, img_files[i])).astype(float)
    amp = imageio.imread(os.path.join(data_path, img_files[i + 1])).astype(float)

    i1, i2, ph0 = generate_data(ph, amp, delta_ph_max, z1, z2, lambda_, dx, nx, ny, px, py)

    if save_as_h5:
        with h5py.File(h5_file, 'a') as h5f:
            for dataset_name, data in zip(['inputs', 'targets', 'phase0'], [i1, i2, ph0]):
                dset = h5f[dataset_name]
                dset.resize((dset.shape[0] + 1), axis=0)
                dset[-1] = data[..., np.newaxis]

    print(f"{i+1}/{img_count}")

end_time = time.time()
print(" ")
print(f"Czas wykonania: {end_time - start_time:.1f} [s]")


if save_as_h5:
    h5f.close()
    print(f"\nData has been saved as: {h5_file}")

if save_as_h5 and save_disc_space:
    print("Please wait to compress files...")
    with h5py.File(h5_file, 'r') as f_in:
        with h5py.File(f'{output_file}_compressed.h5', 'w') as f_out:
            for dataset_name in f_in:
                data = f_in[dataset_name][:]

                f_out.create_dataset(dataset_name, data=data, compression='gzip', compression_opts=9, dtype=data.dtype)

    print(f"File '{output_file}.h5' has been compressed.")
