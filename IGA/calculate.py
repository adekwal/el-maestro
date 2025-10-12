import numpy as np
from skimage.metrics import structural_similarity as ssim

def calc_RMSE(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Different sizes")

    mse = np.mean((image1 - image2) ** 2)
    return np.round(np.sqrt(mse), 3)

def calc_SSIM(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Different sizes")

    ssim_value = ssim(image1, image2, data_range=image1.max() - image1.min())
    return np.round(ssim_value, 3)