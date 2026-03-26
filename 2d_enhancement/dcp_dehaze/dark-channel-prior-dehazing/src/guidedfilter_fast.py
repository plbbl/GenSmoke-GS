#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast guided filter (box-filter formulation, luminance guidance) for refining transmission."""

import cv2
import numpy as np


def guided_filter(I, p, r=40, eps=1e-3):
    """Refine scalar map `p` using RGB image `I` as guidance.

    I: MxNx3 float, typically in [0, 1]
    p: MxN transmission map
    r: radius (window size = 2r+1)
    """
    if I.ndim != 3 or I.shape[2] != 3:
        raise ValueError("I must be MxNx3 RGB")
    p = np.asarray(p, dtype=np.float64)
    bgr = (np.clip(I, 0.0, 1.0) * 255.0).astype(np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0

    k = 2 * int(r) + 1
    mean_I = cv2.boxFilter(gray, cv2.CV_64F, (k, k))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (k, k))
    mean_Ip = cv2.boxFilter(gray * p, cv2.CV_64F, (k, k))
    mean_II = cv2.boxFilter(gray * gray, cv2.CV_64F, (k, k))
    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (k, k))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (k, k))
    return mean_a * gray + mean_b
