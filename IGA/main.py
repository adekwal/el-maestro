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
    h5_file_path = r"C:\Users\Administrator\Desktop\adek files\python\dataset\test_images_databases\pw_512x512.h5"

model_path = r"C:\Users\Administrator\Desktop\adek files\python\dataset\modele ResNet\model_50_official.keras" # path to trained ResNet model

# Simulation parameters
lambda_ = 0.561
dx = 2.4                # [um]
delta_z = 8.2222e3      # [um]
z1 = 3.5578e3
z2 = z1 + delta_z
z = np.array([z1, z2])
sigma = 2
iterations = 5

# Different labels
if dane_testowe_kulki:
    I1, I2, ph0, Nx, Ny, zz1, zz2 = load_data_kulki(h5_file_path)
else:
    I1, I2, ph0, Nx, Ny = load_data_mod(h5_file_path)

# # Squeeze
# I1 = np.squeeze(I1)
# I2 = np.squeeze(I2)

# Generating image
model = tf.keras.models.load_model(model_path) # load the trained ResNet model
I2_predicted = model.predict(I1[np.newaxis, :, :, np.newaxis])[0, :, :, 0]
print("Image has been generated")
print("Please wait...")

I1 = I1 / np.max(I1)
I2 = I2 / np.max(I2)
I2_predicted = I2_predicted / np.max(I2_predicted)

H = np.stack([I1, I2], axis=-1)
H_AI = np.stack([I1, I2_predicted], axis=-1)

# Plot intensity
i = [I1, I2, I2_predicted]
i_tit = ['I1', 'I2', 'I2_predicted']
plot_charts(i, i_tit)


# Optical field at z0
R_GS  = IGA(H, z, lambda_, dx, 0, iterations)
R_GSai = IGA(H_AI, z, lambda_, dx, 0, iterations)
R_IGA = IGA(H_AI, z, lambda_, dx, sigma, iterations)
R_GA  = IGA(H, z, lambda_, dx, np.inf, iterations)
R_GAai  = IGA(H_AI, z, lambda_, dx, np.inf, iterations)

# Charts
ph = [np.angle(R_GS), np.angle(R_GSai), np.angle(R_GA), np.angle(R_GAai), np.angle(R_IGA), ph0]
ph_tit = ["GS [i1, i2]", "GSai [i1, i2_pred]", "GA [i1, i2]", "GAai [i1, i2_pred]", "IGA(sigma=2, iter=5)", "ph0"]
plot_charts(ph, ph_tit, suptitle='Reconstructed ph0 comparison', cbar_label='phase [rad]')

# Difference with GS
diff_GSai = np.angle(R_GSai) - np.angle(R_GS)
diff_GA = np.angle(R_GA) - np.angle(R_GS)
diff_GAai = np.angle(R_GAai) - np.angle(R_GS)
diff_IGA = np.angle(R_IGA) - np.angle(R_GS)

df = [diff_GSai, diff_GA, diff_GAai, diff_IGA]
df_tit = ["GSai", "GA", "GAai", "IGA"]
plot_charts(df, df_tit, suptitle="Difference of reconstructed ph0 and GS[i1, i2] comparison", cbar_label='phase [rad]')

# More detailed comparison
df2 = [diff_GSai, diff_IGA]
df2_tit = ["GSai", "IGA"]
plot_charts(df2, df2_tit, suptitle="Difference of reconstructed ph0 and GS[i1, i2] comparison", cbar_label='phase [rad]')

# Calculate RMSE & SSIM to GS
results = [
    ["GS &", "RMSE", "SSIM"],
    ["GSai", calc_RMSE(np.angle(R_GSai), np.angle(R_GS)), calc_SSIM(np.angle(R_GSai), np.angle(R_GS))],
    ["GA", calc_RMSE(np.angle(R_GA), np.angle(R_GS)), calc_SSIM(np.angle(R_GA), np.angle(R_GS))],
    ["GAai", calc_RMSE(np.angle(R_GAai), np.angle(R_GS)), calc_SSIM(np.angle(R_GAai), np.angle(R_GS))],
    ["IGA", calc_RMSE(np.angle(R_IGA), np.angle(R_GS)), calc_SSIM(np.angle(R_IGA), np.angle(R_GS))]
]
results_table = tabulate(results, tablefmt="rounded_grid")
tf.print(results_table)