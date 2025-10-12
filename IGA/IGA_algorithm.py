import numpy as np
from scipy.ndimage import gaussian_filter
from propagate import propagate_plane_wave

def IGA(H, z, lambda_, dx, sigma=2, iter=5):
    Ny, Nx, N = H.shape
    dy = dx
    lambda_arr = np.full(N, lambda_) if np.isscalar(lambda_) else np.array(lambda_)

    if sigma < np.inf:
        if sigma > 0:
            Hblurred = np.zeros_like(H)
            for n in range(N):
                Hblurred[:, :, n] = gaussian_filter(H[:, :, n], sigma)
        else:
            Hblurred = H.copy()

        # GS reconstruction
        if len(lambda_arr) == 1:
            R_GS = GS_multiHeight(Hblurred, z, lambda_arr[0], dx, dy, iter)
        else:
            R_GS = GS_multiWavelength(Hblurred, z, lambda_arr, dx, dy, iter)

    # GA reconstruction
    if sigma > 0:
        R_GA = GA(H, z, lambda_arr, dx, dy)

    # Final combination
    if sigma == 0:
        return R_GS
    elif sigma == np.inf:
        return R_GA
    else:
        Rea = np.real(R_GS) + np.real(R_GA) - gaussian_filter(np.real(R_GA), sigma)
        Ima = np.imag(R_GS) + np.imag(R_GA) - gaussian_filter(np.imag(R_GA), sigma)
        return Rea + 1j * Ima


def GS_multiHeight(H, z, lambda_, dx, dy, iter):
    A = np.sqrt(H)
    U = A[:, :, 0].astype(np.complex128)

    for _ in range(iter):
        for n in range(len(z) - 1):
            U = propagate_plane_wave(U, z[n + 1] - z[n], 1.0, lambda_, dx, dy)
            U = U / np.abs(U) * A[:, :, n + 1]
        U = propagate_plane_wave(U, z[0] - z[-1], 1.0, lambda_, dx, dy)
        U = U / np.abs(U) * A[:, :, 0]

    return propagate_plane_wave(U, -z[0], 1.0, lambda_, dx, dy)


def GS_multiWavelength(H, z, lambda_arr, dx, dy, iter):
    A = np.sqrt(H)
    U = A[:, :, 0].astype(np.complex128)

    for _ in range(iter):
        for n in range(len(z) - 1):
            R = propagate_plane_wave(U, -z[n], 1.0, lambda_arr[n], dx, dy)
            phase = np.angle(R) * lambda_arr[n] / lambda_arr[n + 1]
            R = np.abs(R) * np.exp(1j * phase)
            U = propagate_plane_wave(R, z[n + 1], 1.0, lambda_arr[n + 1], dx, dy)
            U = U / np.abs(U) * A[:, :, n + 1]

        R = propagate_plane_wave(U, -z[-1], 1.0, lambda_arr[-1], dx, dy)
        phase = np.angle(R) * lambda_arr[-1] / lambda_arr[0]
        R = np.abs(R) * np.exp(1j * phase)
        U = propagate_plane_wave(R, z[0], 1.0, lambda_arr[0], dx, dy)
        U = U / np.abs(U) * A[:, :, 0]

    return propagate_plane_wave(U, -z[0], 1.0, lambda_arr[0], dx, dy)


def GA(H, z, lambda_arr, dx, dy):
    Ny, Nx, N = H.shape
    R = np.zeros((Ny, Nx), dtype=np.complex128)

    for n in range(N):
        U = propagate_plane_wave(H[:, :, n], -z[n], 1.0, lambda_arr[n], dx, dy)
        phase = np.angle(U) * lambda_arr[n] / lambda_arr[0]
        R += np.sqrt(np.abs(U)) * np.exp(1j * phase)

    return R / N