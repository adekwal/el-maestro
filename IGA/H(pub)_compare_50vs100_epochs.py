import numpy as np
import tensorflow as tf
from IGA_algorithm import IGA
from unpack import load_data, load_data_mod, load_data_kulki
from plots import plot_charts
from calculate import calc_RMSE, calc_SSIM
from tabulate import tabulate

# User parameters
dane_testowe_kulki = False
if dane_testowe_kulki:
    h5_file_path = r"C:\Users\Administrator\Desktop\adek files\python\dataset\dane_testowe\kulki.h5"
else:
    h5_file_path = r"C:\Users\Administrator\Desktop\adek files\python\dataset\test_images_databases\iss_256x256.h5"

model_50_path = r"C:\Users\Administrator\Desktop\adek files\python\dataset\modele ResNet\model_4000_256x256_50ep.keras"
model_100_path = r"C:\Users\Administrator\Desktop\adek files\python\dataset\modele ResNet\model_4000_256x256_100ep.keras"

# Simulation parameters
lambda_ = 0.561
dx = 2.4                # [um]
delta_z = 8.2222e3      # [um]
z1 = 3.5578e3
z2 = z1 + delta_z
z = np.array([z1, z2])
sigma = 10
iterations = 20

# Different labels
if dane_testowe_kulki:
    I1, I2, ph0, Nx, Ny, zz1, zz2 = load_data_kulki(h5_file_path)
else:
    I1, I2, ph0, Nx, Ny = load_data_mod(h5_file_path)

# # Squeeze
# I1 = np.squeeze(I1)
# I2 = np.squeeze(I2)

# Generating image
model_50 = tf.keras.models.load_model(model_50_path)
I2_predicted_50 = model_50.predict(I1[np.newaxis, :, :, np.newaxis])[0, :, :, 0]
print("Image has been generated 1/2")
print("Please wait...")

model_100 = tf.keras.models.load_model(model_100_path)
I2_predicted_100 = model_100.predict(I1[np.newaxis, :, :, np.newaxis])[0, :, :, 0]
print("Image has been generated 2/2")
print("Please wait...")

I1 = I1 / np.max(I1)
I2 = I2 / np.max(I2)
I2_predicted_50 = I2_predicted_50 / np.max(I2_predicted_50)
I2_predicted_100 = I2_predicted_100 / np.max(I2_predicted_100)

# H = np.stack([I1, I2], axis=-1)
H = np.stack([I1, I2], axis=-1)
H_50 = np.stack([I1, I2_predicted_50], axis=-1)
H_100 = np.stack([I1, I2_predicted_100], axis=-1)

# # Plot intensity
# i = [I1, I2, I2_predicted]
# i_tit = ['I1', 'I2', 'I2_predicted']
# plot_charts(i, i_tit)


# Optical field at z0
R_GSai = IGA(H, z, lambda_, dx, 0, iterations)
R_IGA_50 = IGA(H_50, z, lambda_, dx, sigma, iterations)
R_IGA_100 = IGA(H_100, z, lambda_, dx, sigma, iterations)

# Charts
ph = [np.angle(R_IGA_50), np.angle(R_IGA_100)]
ph_tit = ["ph0 [ResNet 50 epochs]", "ph0 [ResNet 100 epochs]"]
plot_charts(ph, ph_tit, suptitle='Reconstructed ph0 comparison with IGA(sigma=2, iter=5)', cbar_label='phase [rad]')


# Calculate RMSE & SSIM to GS
results = [
    ["GS &", "RMSE", "SSIM"],
    ["IGA_50", calc_RMSE(np.angle(R_IGA_50), np.angle(R_GSai)), calc_SSIM(np.angle(R_IGA_50), np.angle(R_GSai))],
    ["IGA_100", calc_RMSE(np.angle(R_IGA_100), np.angle(R_GSai)), calc_SSIM(np.angle(R_IGA_100), np.angle(R_GSai))]
]
results_table = tabulate(results, tablefmt="rounded_grid")
tf.print(results_table)