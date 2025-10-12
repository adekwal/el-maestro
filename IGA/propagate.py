import numpy as np

def propagate_plane_wave(uin, z, n0, lambda_, dx, dy):
    Ny, Nx = uin.shape
    k = 2 * np.pi / lambda_

    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)

    Fx, Fy = np.meshgrid(fx, fy)

    if z < 0:
        uin = np.conj(uin)

    FTu = np.fft.fft2(uin)

    # Angular spectrum kernel
    kernel = np.exp(1j * k * np.abs(z) * np.real(np.sqrt(n0**2 - lambda_**2 * (Fx**2 + Fy**2))))
    kernel[n0**2 < lambda_**2 * (Fx**2 + Fy**2)] = 0

    FTu *= kernel

    uout = np.fft.ifft2(FTu)

    if z < 0:
        uout = np.conj(uout)

    return uout
