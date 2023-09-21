import numpy as np
import numpy.fft as fft


def spatial_bandpass_filter(img, min_freq, max_freq):
    """
    Filters a 2D array to retain only spatial frequencies within a given range.
    
    Parameters:
    - img: 2D numpy array (e.g., grayscale image)
    - min_freq: minimum spatial frequency to retain
    - max_freq: maximum spatial frequency to retain
    
    Returns:
    - Filtered 2D numpy array
    """
    # Compute the 2D FFT of the image
    f = fft.fftshift(fft.fft2(img))
    
    # Create a frequency grid
    rows, cols = img.shape
    cy, cx = rows // 2, cols // 2  # center coordinates
    y = np.linspace(-cy, cy, rows)
    x = np.linspace(-cx, cx, cols)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)  # Radius matrix
    
    # Create a mask that keeps only frequencies within the desired range
    mask = (R >= min_freq) & (R <= max_freq)

    # Apply the mask
    f_filtered = f * mask

    # Inverse FFT to get the filtered image
    img_filtered = np.real(fft.ifft2(fft.ifftshift(f_filtered)))
    
    return img_filtered