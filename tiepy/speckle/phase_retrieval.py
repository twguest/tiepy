import numpy as np

from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2


from math import pi as pi
from math import floor as floor

from tqdm import tqdm

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

def paganin_algorithm(ii, z, wav, delta, beta):
    """
    Paganin Algorithm for phase retrieval.

    :param ii: numpy.ndarray
        4D array of shape (Nx, Ny, K, N), representing the set of K by N projection images.
        Nx and Ny are the dimensions of each image, K is the number of projection angles,
        and N is the number of iterations.
    :param z: float
        Propagation distance. The distance between the object and the detector.
    :param wav: float
        X-ray wavelength.
    :param delta: float
        Refractive index decrement.
    :param beta: float
        X-ray attenuation coefficient.

    :return: numpy.ndarray
        4D array of shape (Nx, Ny, K, N), containing the estimated phase for each projection image.

    :raises:
        AssertionError: If the input 'ii' does not have dtype 'float64'.

    :Notes:
        - The input ii should have dtype 'float64'.
        - The output phase represents the estimated phase shift caused by the object.

    :Reference:
        D. Paganin, "Simultaneous phase and amplitude extraction from a single defocused image of a homogeneous object,"
        Journal of Microscopy, vol. 206, no. 1, pp. 33-40, 2002.
    """


    assert ii.dtype == "float64"

    phase = np.zeros_like(ii)

    Nx = ii.shape[0]
    Ny = ii.shape[1]

    flatfield = np.ones([Nx, Ny])
    flatfield /= np.sum(flatfield)

    dkx = 2 * np.pi / Nx
    dky = 2 * np.pi / Ny

    kx, ky = np.meshgrid(
        (np.arange(0, Ny) - np.floor(Ny / 2) - 1) * dky,
        (np.arange(0, Nx) - np.floor(Nx / 2) - 1) * dkx,
    )

    filtre = 1 + (wav * z * delta * 4 * (np.pi**2) * (kx**2 + ky**2) / (4 * np.pi * beta))

    for itr in tqdm(range(ii.shape[-1])):
        for k in range(ii.shape[-2]):
            i1 = ii[:, :, k, itr]
            i1 /= np.sum(i1)
            i1 -= np.mean(i1)

            trans_func = np.log(np.real(np.fft.ifft2(np.fft.fft2(i1 / flatfield) / filtre)))
            trans_func[np.isnan(trans_func)] = 0

            phase[:, :, k, itr] = (delta / (2 * beta)) * trans_func

    return phase