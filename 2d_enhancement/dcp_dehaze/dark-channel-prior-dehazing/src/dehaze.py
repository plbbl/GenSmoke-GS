#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implementation for Single Image Haze Removal Using Dark Channel Prior.

Reference:
http://research.microsoft.com/en-us/um/people/kahe/cvpr09/
http://research.microsoft.com/en-us/um/people/kahe/eccv10/
"""

import cv2
import numpy as np
from PIL import Image

from guidedfilter_fast import guided_filter

R, G, B = 0, 1, 2  # index for convenience
L = 256  # color depth


def get_dark_channel(I, w):
    """Get the dark channel prior in the (RGB) image data.

    Parameters
    -----------
    I:  an M * N * 3 numpy array containing data ([0, L-1]) in the image where
        M is the height, N is the width, 3 represents R/G/B channels.
    w:  window size

    Return
    -----------
    An M * N array for the dark channel prior ([0, L-1]).
    """
    _m, _n, _ = I.shape
    # CVPR09 Eq.5: per-channel min filter (wxw), then min across RGB
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w), int(w)))
    er = [cv2.erode(I[:, :, c].astype(np.float64), kernel) for c in range(3)]
    return np.minimum(np.minimum(er[0], er[1]), er[2])


def get_atmosphere(I, darkch, p):
    """Get the atmosphere light in the (RGB) image data.

    Parameters
    -----------
    I:      the M * N * 3 RGB image data ([0, L-1]) as numpy array
    darkch: the dark channel prior of the image as an M * N numpy array
    p:      percentage of pixels for estimating the atmosphere light

    Return
    -----------
    A 3-element array containing atmosphere light ([0, L-1]) for each channel
    """
    # reference CVPR09, 4.4
    M, N = darkch.shape
    flatI = I.reshape(M * N, 3)
    flatdark = darkch.ravel()
    n_pick = max(1, int(M * N * p))
    searchidx = (-flatdark).argsort()[:n_pick]  # brightest in dark channel, CVPR09 §4.4

    # return the highest intensity for each channel
    return np.max(flatI.take(searchidx, axis=0), axis=0)


def get_transmission(I, A, darkch, omega, w):
    """Get the transmission esitmate in the (RGB) image data.

    Parameters
    -----------
    I:       the M * N * 3 RGB image data ([0, L-1]) as numpy array
    A:       a 3-element array containing atmosphere light
             ([0, L-1]) for each channel
    darkch:  the dark channel prior of the image as an M * N numpy array
    omega:   bias for the estimate
    w:       window size for the estimate

    Return
    -----------
    An M * N array containing the transmission rate ([0.0, 1.0])
    """
    return 1 - omega * get_dark_channel(I / A, w)  # CVPR09, eq.12


def dehaze_raw(I, tmin=0.2, Amax=220, w=15, p=0.0001,
               omega=0.95, guided=True, r=40, eps=1e-3):
    """Get the dark channel prior, atmosphere light, transmission rate
       and refined transmission rate for raw RGB image data.

    Parameters
    -----------
    I:      M * N * 3 data as numpy array for the hazy image
    tmin:   threshold of transmission rate
    Amax:   threshold of atmosphere light
    w:      window size of the dark channel prior
    p:      percentage of pixels for estimating the atmosphere light
    omega:  bias for the transmission estimate

    guided: whether to use the guided filter to fine the image
    r:      the radius of the guidance
    eps:    epsilon for the guided filter

    Return
    -----------
    (Idark, A, rawt, refinedt) if guided=False, then rawt == refinedt
    """
    m, n, _ = I.shape
    Idark = get_dark_channel(I, w)

    A = get_atmosphere(I, Idark, p)
    A = np.minimum(A, Amax)  # threshold A

    rawt = get_transmission(I, A, Idark, omega, w)

    rawt = refinedt = np.maximum(rawt, tmin)  # threshold t
    if guided:
        rng = float(I.max() - I.min())
        if rng < 1e-12:
            normI = np.zeros_like(I)
        else:
            normI = (I - I.min()) / rng  # normalize I for guidance
        refinedt = guided_filter(normI, refinedt, r, eps)

    return Idark, A, rawt, refinedt


def get_radiance(I, A, t):
    """Recover the radiance from raw image data with atmosphere light
       and transmission rate estimate.

    Parameters
    ----------
    I:      M * N * 3 data as numpy array for the hazy image
    A:      a 3-element array containing atmosphere light
            ([0, L-1]) for each channel
    t:      estimate fothe transmission rate

    Return
    ----------
    M * N * 3 numpy array for the recovered radiance
    """
    tiledt = np.zeros_like(I)  # tiled to M * N * 3
    tiledt[:, :, R] = tiledt[:, :, G] = tiledt[:, :, B] = t
    return (I - A) / tiledt + A  # CVPR09, eq.16


def dehaze(im, tmin=0.2, Amax=220, w=15, p=0.0001,
           omega=0.95, guided=True, r=40, eps=1e-3):
    """Dehaze the given RGB image.

    Parameters
    ----------
    im:     the Image object of the RGB image
    guided: refine the dehazing with guided filter or not
    other parameters are the same as `dehaze_raw`

    Return
    ----------
    (dark, rawt, refinedt, rawrad, rerad)
    Images for dark channel prior, raw transmission estimate,
    refiend transmission estimate, recovered radiance with raw t,
    recovered radiance with refined t.
    """
    I = np.asarray(im, dtype=np.float64)
    Idark, A, rawt, refinedt = dehaze_raw(I, tmin, Amax, w, p,
                                          omega, guided, r, eps)
    white = np.full_like(Idark, L - 1)

    def to_img(raw):
        # threshold to [0, L-1]
        cut = np.maximum(np.minimum(raw, L - 1), 0).astype(np.uint8)

        if len(raw.shape) == 3:
            return Image.fromarray(cut)
        else:
            return Image.fromarray(cut)

    return [to_img(raw) for raw in (Idark, white * rawt, white * refinedt,
                                    get_radiance(I, A, rawt),
                                    get_radiance(I, A, refinedt))]
