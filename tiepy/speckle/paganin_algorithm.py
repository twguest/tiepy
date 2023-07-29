"""
exec paganin algorithm 
"""

import numpy as np
from tqdm import tqdm


def paganin_algorithm(ii, z, wav, delta, beta):
    """
    Paganin Algorithm for phase retrieval.

    This function applies the Paganin Algorithm to perform phase retrieval on a given set of projection images.
    The algorithm is used to estimate the phase shift caused by an object based on its measured intensity
    and some system parameters.

    Parameters:
        ii (numpy.ndarray): 4D array of shape (Nx, Ny, K, N), representing the set of K by N projection images.
                            Nx and Ny are the dimensions of each image, K is the number of projection angles,
                            and N is the number of iterations.
        z (float): Propagation distance. The distance between the object and the detector. [float]
        wav (float): X-ray wavelength. [float]
        delta (float): Refractive index decrement. [float]
        beta (float): X-ray attenuation coefficient. [float]

    Returns:
        numpy.ndarray: 4D array of shape (Nx, Ny, K, N), containing the estimated phase for each projection image.

    Notes:
        - The input ii should have dtype 'float64'.
        - The output phase represents the estimated phase shift caused by the object.

    Reference:
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


if __name__ == "__main__":
    import sys
    import h5py as h5

    from felpy.utils.opt_utils import ekev2wav

    print(sys.argv)

    with h5.File(sys.argv[1], "r") as hf:
        ii = hf.get("data")[()].T

    z = float(sys.argv[2])
    wav = ekev2wav(float(sys.argv[3]))
    delta = float(sys.argv[4])
    beta = float(sys.argv[5])

    paganin_phase = paganin_algorithm(ii, z, wav, delta, beta)

    with h5.File(sys.argv[6], "w") as hf:
        hf.create_dataset(name="data", data=paganin_phase)
