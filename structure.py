import numpy as np
import scipy.ndimage as ndimage
import pylab as plt

eps = np.finfo(np.float).eps

def structure_tensor(im, radius):
    bi = ndimage.gaussian_filter(im.astype(np.float32), 1)
    imy = ndimage.sobel(bi, axis=0)
    imx = ndimage.sobel(bi, axis=1)
    Axx = ndimage.gaussian_filter(imx ** 2, radius)
    Axy = ndimage.gaussian_filter(imx * imy, radius)
    Ayy = ndimage.gaussian_filter(imy ** 2, radius)
    return Axx, Axy, Ayy

def orientation_field(im, radius):
    Axx, Axy, Ayy = structure_tensor(im, radius)
    T = Axx + Ayy
    D = Axx * Ayy - Axy ** 2
    tmp = np.sqrt(4 * (Axy ** 2) + (Axx - Ayy) ** 2)
    tmp[np.isnan(tmp)] = 0
    e1 = (T + tmp) / 2
    e2 = (T - tmp) / 2

    coherence = (e1 - e2) / (e1 + e2 + eps)
    assert np.all(~np.isnan(coherence))
    eig1x = Axy
    eig1y = e1 - Axx
    n = np.sqrt(eig1x ** 2 + eig1y ** 2 + eps)
    return coherence, eig1x / n, eig1y / n
