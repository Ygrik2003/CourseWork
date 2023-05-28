"""
Methods to characterize image textures.
"""

import numpy as np
from math import sin, cos
import numba as nb


# @nb.jit
# def _glcm_loop_py(image, distances,
#                angles, levels,
#                out):
#     """Perform co-occurrence matrix accumulation.

#     Parameters
#     ----------
#     image : ndarray
#         Integer typed input image. Only positive valued images are supported.
#         If type is other than uint8, the argument `levels` needs to be set.
#     distances : ndarray
#         List of pixel pair distance offsets.
#     angles : ndarray
#         List of pixel pair angles in radians.
#     levels : int
#         The input image should contain integers in [0, `levels`-1],
#         where levels indicate the number of gray-levels counted
#         (typically 256 for an 8-bit image).
#     out : ndarray
#         On input a 4D array of zeros, and on output it contains
#         the results of the GLCM computation.

#     """
#     rows = image.shape[0]
#     cols = image.shape[1]

#     for a_idx in range(angles.shape[0]):
#         angle = angles[a_idx]
#         for d_idx in range(distances.shape[0]):
#             distance = distances[d_idx]
#             offset_row = round(sin(angle) * distance)
#             offset_col = round(cos(angle) * distance)
#             start_row = max(0, -offset_row)
#             end_row = min(rows, rows - offset_row)
#             start_col = max(0, -offset_col)
#             end_col = min(cols, cols - offset_col)
#             for r in range(start_row, end_row):
#                 for c in range(start_col, end_col):
#                     i = image[r, c]
#                     # compute the location of the offset pixel
#                     row = r + offset_row
#                     col = c + offset_col
#                     j = image[row, col]
#                     if 0 <= i < levels and 0 <= j < levels:
#                         out[i, j, d_idx, a_idx] += 1

@nb.njit
def _glcm_loop_py(image, offset_row,
               offset_col, levels):
    lastIndex = 0
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
    rows = image.shape[0]
    cols = image.shape[1]

    start_row = max(0, -offset_row)
    end_row = min(rows, rows - offset_row)
    start_col = max(0, -offset_col)
    end_col = min(cols, cols - offset_col)

    out = np.zeros((np.unique(image).shape[0] ** 2, 3), dtype=np.float32)

    for r in range(start_row, end_row):
        for c in range(start_col, end_col):
            i = image[r, c]
            # compute the location of the offset pixel
            row = r + offset_row
            col = c + offset_col
            j = image[row, col]
            if 0 <= i < levels and 0 <= j < levels:
                # if not np.any(np.all([out[:, 0] == i, out[:, 1] == j], axis=0)):
                #     out = np.append(out, [[i, j, 0]], axis=0)

                # out[np.all([out[:, 0] == i, out[:, 1] == j], axis=0), 2] += 1
                for i_elem in range(out.shape[0]):
                    if out[i_elem][0] == i and out[i_elem][1] == j:
                        out[i_elem][2] = out[i_elem][2] + 1
                        break
                else:
                    out[lastIndex] = [i, j, 1]
                    lastIndex += 1
    return out[:lastIndex]

@nb.njit
def graycomatrix(image, offset_row, offset_col, levels=None, symmetric=False,
                 normed=False):
    """Calculate the gray-level co-occurrence matrix.

    A gray level co-occurrence matrix is a histogram of co-occurring
    grayscale values at a given offset over an image.

    Parameters
    ----------
    image : array_like
        Integer typed input image. Only positive valued images are supported.
        If type is other than uint8, the argument `levels` needs to be set.
    distances : array_like
        List of pixel pair distance offsets.
    angles : array_like
        List of pixel pair angles in radians.
    levels : int, optional
        The input image should contain integers in [0, `levels`-1],
        where levels indicate the number of gray-levels counted
        (typically 256 for an 8-bit image). This argument is required for
        16-bit images or higher and is typically the maximum of the image.
        As the output matrix is at least `levels` x `levels`, it might
        be preferable to use binning of the input image rather than
        large values for `levels`.
    symmetric : bool, optional
        If True, the output matrix `P[:, :, d, theta]` is symmetric. This
        is accomplished by ignoring the order of value pairs, so both
        (i, j) and (j, i) are accumulated when (i, j) is encountered
        for a given offset. The default is False.
    normed : bool, optional
        If True, normalize each matrix `P[:, :, d, theta]` by dividing
        by the total number of accumulated co-occurrences for the given
        offset. The elements of the resulting matrix sum to 1. The
        default is False.

    Returns
    -------
    P : 4-D ndarray
        The gray-level co-occurrence histogram. The value
        `P[i,j,d,theta]` is the number of times that gray-level `j`
        occurs at a distance `d` and at an angle `theta` from
        gray-level `i`. If `normed` is `False`, the output is of
        type uint32, otherwise it is float64. The dimensions are:
        levels x levels x number of distances x number of angles.

    References
    ----------
    .. [1] M. Hall-Beyer, 2007. GLCM Texture: A Tutorial
           https://prism.ucalgary.ca/handle/1880/51900
           DOI:`10.11575/PRISM/33280`
    .. [2] R.M. Haralick, K. Shanmugam, and I. Dinstein, "Textural features for
           image classification", IEEE Transactions on Systems, Man, and
           Cybernetics, vol. SMC-3, no. 6, pp. 610-621, Nov. 1973.
           :DOI:`10.1109/TSMC.1973.4309314`
    .. [3] M. Nadler and E.P. Smith, Pattern Recognition Engineering,
           Wiley-Interscience, 1993.
    .. [4] Wikipedia, https://en.wikipedia.org/wiki/Co-occurrence_matrix


    Examples
    --------
    Compute 2 GLCMs: One for a 1-pixel offset to the right, and one
    for a 1-pixel offset upwards.

    >>> image = np.array([[0, 0, 1, 1],
    ...                   [0, 0, 1, 1],
    ...                   [0, 2, 2, 2],
    ...                   [2, 2, 3, 3]], dtype=np.uint8)
    >>> result = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
    ...                       levels=4)
    >>> result[:, :, 0, 0]
    array([[2, 2, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 1],
           [0, 0, 0, 1]], dtype=uint32)
    >>> result[:, :, 0, 1]
    array([[1, 1, 3, 0],
           [0, 1, 1, 0],
           [0, 0, 0, 2],
           [0, 0, 0, 0]], dtype=uint32)
    >>> result[:, :, 0, 2]
    array([[3, 0, 2, 0],
           [0, 2, 2, 0],
           [0, 0, 1, 2],
           [0, 0, 0, 0]], dtype=uint32)
    >>> result[:, :, 0, 3]
    array([[2, 0, 0, 0],
           [1, 1, 2, 0],
           [0, 0, 2, 1],
           [0, 0, 0, 0]], dtype=uint32)

    """

    image = np.ascontiguousarray(image)

    image_max = image.max()

    if levels is None:
        levels = 256

    if image_max >= levels:
        raise ValueError("The maximum grayscale value in the image should be "
                         "smaller than the number of levels.")

       
    # count co-occurences
    P = _glcm_loop_py(image, offset_row, offset_col, levels)

    # make each GLMC symmetric
    if symmetric:
        Pt = np.transpose(P, (1, 0))
        P = P + Pt

    # normalize each GLCM
    if normed:
        glcm_sums = np.sum(P[:, 2])
        glcm_sums = 1 if glcm_sums == 0 else glcm_sums
        P[:, 2] /= glcm_sums
    return P