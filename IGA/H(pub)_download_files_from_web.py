from downloader import download_file, download_from_drive
import tensorflow as tf
import numpy as np
# import gdown
from IGA_algorithm import IGA
from unpack import load_data_mod
from plots import plot_charts
from calculate import calc_RMSE, calc_SSIM
from tabulate import tabulate
import os

### User setup ###

dest_folder = r"C:\Users\Administrator\Desktop\adek files\python\IGA"   # paste here your destination path
resnet_filename = "model_4000_256x256_50ep.keras"                       # do not change yet
images_filename = "pw_512x512.h5"                                       # do not change yet

###


# Simulation parameters
lambda_ = 0.561
dx = 2.4                # [um]
delta_z = 8.2222e3      # [um]
z1 = 3.5578e3
z2 = z1 + delta_z
z = np.array([z1, z2])
sigma = 10
iterations = 5


# URL addresses to files on Github
model_url = os.path.join("https://raw.githubusercontent.com/adekwal/el-maestro/main/_storage/", resnet_filename)
model_path = download_file(model_url, dest_folder)
h5_file_url = "https://drive.google.com/file/d/1XCzFs-Q_3DnageeJmCqbWPupEmgduzOT/view?usp=drive_link"
# h5_file_path = gdown.download("https://drive.google.com/uc?id=1XCzFs-Q_3DnageeJmCqbWPupEmgduzOT", os.path.join(dest_folder, images_filename), quiet=False)
h5_file_path = download_from_drive(h5_file_url, dest_folder, images_filename)


# Loading data from .h5 file
I1, I2, ph0, Nx, Ny = load_data_mod(h5_file_path)


# Generating image
model = tf.keras.models.load_model(model_path)
print("ResNet model has been loaded")
I2_predicted = model.predict(I1[np.newaxis, :, :, np.newaxis])[0, :, :, 0]
print("Image has been generated")
print("Please wait...")


# Normalization
I1 = I1 / np.max(I1)
I2 = I2 / np.max(I2)
I2_predicted = I2_predicted / np.max(I2_predicted)


# Stacking
H = np.stack([I1, I2], axis=-1)
H_AI = np.stack([I1, I2_predicted], axis=-1)


# Optical field at z0
R_GS  = IGA(H, z, lambda_, dx, 0, iterations)
R_GSai = IGA(H_AI, z, lambda_, dx, 0, iterations)
R_IGA = IGA(H_AI, z, lambda_, dx, sigma, iterations)


# Charts
ph = [np.angle(R_GS), np.angle(R_GSai), np.angle(R_IGA), ph0]
ph_tit = ["GS", "GSai", f"IGA(sigma={sigma}, iter={iterations})", "ph0"]
plot_charts(ph, ph_tit, suptitle='Reconstructed ph0 comparison', cbar_label='phase [rad]')


# Calculate RMSE & SSIM to GS
results = [
    ["ph0 &", "RMSE", "SSIM"],
    ["GS", calc_RMSE(np.angle(R_GS), ph0), calc_SSIM(np.angle(R_GS), ph0)],
    ["GSai", calc_RMSE(np.angle(R_GSai), ph0), calc_SSIM(np.angle(R_GSai), ph0)],
    ["IGA", calc_RMSE(np.angle(R_IGA), ph0), calc_SSIM(np.angle(R_IGA), ph0)]
]
results_table = tabulate(results, tablefmt="rounded_grid")
tf.print(results_table)