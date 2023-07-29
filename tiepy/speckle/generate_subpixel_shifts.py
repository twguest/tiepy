import cv2 as cv
import numpy as np
import matplotlib
from tqdm import tqdm
from felpy import Grids
from matplotlib import pyplot as plt
import h5py as h5
import matplotlib
from phenom.utils import e2wav
from tiepy.speckle.template_matching import match_template
from tiepy.speckle.template_utils import get_windows, generate_gaussian_mask
from numpy import fft
from tiepy.speckle.phase_retrieval import kottler

### next step to write it to (npulse x ntrains x 2 (x,y)) dimension array
import matplotlib

crop = 150
windows, c = get_windows(data[:crop, :crop, 0, 0], 50, 4)
print(len(windows))

I = 15  # data.shape[-2]
K = 250  # data.shape[-1] ### SHOULD ASSERT INT OR LIST


loc = np.zeros([len(windows), I, K, 2])
shift = np.zeros([len(windows), I, K, 2])
centers = np.zeros([len(windows), I, K, 2])
itr = 0

for sel in tqdm(range(len(windows))):  ### itr over window center
    j = 0

    for i in range(I):  ### itr over pulse (80 kHz)
        for k in range(K):  ### itr over train (0.1Hz)
            image_crop = data[:crop, :crop, i, k]

            templ = windows[sel]

            mask = generate_gaussian_mask(*image_crop.shape, c[sel][0], c[sel][1], templ.shape[0], 10).astype(
                "float32"
            )

            res, max_loc = match_template(templ=templ, image=image_crop * mask)

            loc[sel, i, k, 0] = max_loc[0]
            loc[sel, i, k, 1] = max_loc[1]

            shift_ = (c[sel][0] - max_loc[0], c[sel][1] - max_loc[1])

            shift[sel, i, k, 0] = shift_[0]
            shift[sel, i, k, 1] = shift_[1]

            centers[sel, i, k, 0] = c[sel][0]
            centers[sel, i, k, 1] = c[sel][1]

    itr += 1
