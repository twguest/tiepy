import numpy as np

from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from numpy.fft import fftfreq as fftfreq

from scipy.ndimage.filters import gaussian_filter
from math import pi as pi
from math import floor as floor


def kottler(dX, dY):
    """
    Perform Kottler phase retrieval on a given complex-valued object.

    This function applies the Kottler method for phase retrieval on a complex-valued object
    represented by the real and imaginary parts dX and dY, respectively. The Kottler method
    uses the Fast Fourier Transform (FFT) to estimate the phase of the object.

    Parameters:
        dX (numpy.ndarray): 2D array representing the real part of the complex object.
                            Shape should be (Nx, Ny), where Nx and Ny are the dimensions of the object.
        dY (numpy.ndarray): 2D array representing the imaginary part of the complex object.
                            Shape should be (Nx, Ny), matching the dimensions of dX.

    Returns:
        numpy.ndarray: 2D array containing the estimated phase of the complex object.

    Notes:
        - The input arrays dX and dY should have the same shape and dtype.
        - The output array phi3 will contain the estimated phase of the complex object.
    """
    i = complex(0, 1)
    Nx, Ny = dX.shape
    dqx = 2 * pi / Nx
    dqy = 2 * pi / Ny
    Qx, Qy = np.meshgrid(
        (np.arange(0, Ny) - floor(Ny / 2) - 1) * dqy, (np.arange(0, Nx) - floor(Nx / 2) - 1) * dqx
    )

    polarAngle = np.arctan2(Qx, Qy)
    ftphi = fftshift(fft2(dX + i * dY)) * np.exp(i * polarAngle)
    ftphi[np.isnan(ftphi)] = 0
    phi3 = ifft2(ifftshift(ftphi))
    return phi3.real
