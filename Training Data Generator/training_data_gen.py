import random
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from propagate import propagate_plane_wave


def generate_data(phase, phase_range, amplitude_range, px_trim_value, z1, z2, lambda_, dx, nx, ny, px, py, amplitude_image=None):
    sigma = 85
    if amplitude_image is None:
        amplitude = np.ones(phase.shape)
    else:
        amplitude = amplitude_image

    # Generation of object wave
    ph_range = random.uniform(phase_range[0], phase_range[1])
    amp_range = random.uniform(amplitude_range[0], amplitude_range[1])
    ph0 = phase[px_trim_value:-px_trim_value, px_trim_value:-px_trim_value]
    amp = amplitude[px_trim_value:-px_trim_value, px_trim_value:-px_trim_value]

    # Scaling and processing phase image
    zoom_factor_ph_x = nx/ph0.shape[1]
    zoom_factor_ph_y = ny/ph0.shape[0]
    if ph0.ndim == 3:
        ph0 = np.mean(ph0, axis=-1)
    ph0 = zoom(ph0, (zoom_factor_ph_y, zoom_factor_ph_x), order=1)

    # Scaling and processing amplitude image
    zoom_factor_amp_x = nx/amp.shape[1]
    zoom_factor_amp_y = ny/amp.shape[0]
    if amp.ndim == 3:
        amp = np.mean(amp, axis=-1)
    amp = zoom(amp, (zoom_factor_amp_y, zoom_factor_amp_x), order=1)

    # High-pass filtering
    ph0 = ph0 - gaussian_filter(ph0, sigma)

    # Image normalization
    ph0 = (ph0 - np.min(ph0)) / (np.max(ph0) - np.min(ph0))
    amp = (amp - np.min(amp)) / (np.max(amp) - np.min(amp))

    # Applying phase shifts
    ph0 = ph0 * ph_range

    # Adding padding
    pad_x = (int(py/2-ny/2), int(py/2-ny/2))
    pad_y = (int(px/2-nx/2), int(px/2-nx/2))
    ph0 = np.pad(ph0, (pad_x, pad_y), mode='edge')
    amp = np.pad(amp, (pad_x, pad_y), mode='edge')

    # Calculation of the optical field
    ph_obj = ph0 - np.median(ph0)
    amp_obj = np.ones(ph_obj.shape) if amplitude_image is None else 1 + amp_range*(amp - 0.5)
    u_obj = amp_obj * np.exp(1j * ph_obj)

    # Simulation of two defocused intensity measurements
    u1 = propagate_plane_wave(u_obj, z1, 1, lambda_, dx, dx)
    u2 = propagate_plane_wave(u_obj, z2, 1, lambda_, dx, dx)

    # Calculation of intensity and phase
    i1 = np.abs(u1)**2
    i2 = np.abs(u2)**2

    # Cropping and storing data
    i1 = i1[int(py / 2 - ny / 2): int(py / 2 + ny / 2), int(px / 2 - nx / 2): int(px / 2 + nx / 2)]
    i2 = i2[int(py / 2 - ny / 2): int(py / 2 + ny / 2), int(px / 2 - nx / 2): int(px / 2 + nx / 2)]
    ph_obj = ph_obj[int(py / 2 - ny / 2): int(py / 2 + ny / 2), int(px / 2 - nx / 2): int(px / 2 + nx / 2)]

    return i1, i2, ph_obj
