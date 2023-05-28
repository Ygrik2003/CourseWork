#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as cnp
from libc.math cimport sin, cos, abs
from .._shared.interpolation cimport bilinear_interpolation, round
from .._shared.transform cimport integrate

cdef extern from "numpy/npy_math.h":
    double NAN "NPY_NAN"

from .._shared.fused_numerics cimport np_anyint as any_int
from .._shared.fused_numerics cimport np_real_numeric

cnp.import_array()

def _glcm_loop(any_int[:, ::1] image, double[:] distances,
               double[:] angles, Py_ssize_t levels,
               cnp.uint8_t[:, :, :, ::1] out):
    """Perform co-occurrence matrix accumulation.

    Parameters
    ----------
    image : ndarray
        Integer typed input image. Only positive valued images are supported.
        If type is other than uint8, the argument `levels` needs to be set.
    distances : ndarray
        List of pixel pair distance offsets.
    angles : ndarray
        List of pixel pair angles in radians.
    levels : int
        The input image should contain integers in [0, `levels`-1],
        where levels indicate the number of gray-levels counted
        (typically 256 for an 8-bit image).
    out : ndarray
        On input a 4D array of zeros, and on output it contains
        the results of the GLCM computation.

    """

    cdef:
        Py_ssize_t a_idx, d_idx, r, c, rows, cols, row, col, start_row,\
                   end_row, start_col, end_col, offset_row, offset_col
        any_int i, j
        cnp.float64_t angle, distance

    with nogil:
        rows = image.shape[0]
        cols = image.shape[1]

        for a_idx in range(angles.shape[0]):
            angle = angles[a_idx]
            for d_idx in range(distances.shape[0]):
                distance = distances[d_idx]
                offset_row = round(sin(angle) * distance)
                offset_col = round(cos(angle) * distance)
                start_row = max(0, -offset_row)
                end_row = min(rows, rows - offset_row)
                start_col = max(0, -offset_col)
                end_col = min(cols, cols - offset_col)
                for r in range(start_row, end_row):
                    for c in range(start_col, end_col):
                        i = image[r, c]
                        # compute the location of the offset pixel
                        row = r + offset_row
                        col = c + offset_col
                        j = image[row, col]
                        if 0 <= i < levels and 0 <= j < levels:
                            out[i, j, d_idx, a_idx] += 1
