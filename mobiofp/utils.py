import math

import cv2
import fingerprint_enhancer as fpe
import imutils
import numpy as np
from joblib import Memory

memory = Memory("cache", verbose=0)

"""
Fingertip Processing Functions
"""


def fingertip_enhancement(
    image: np.ndarray,
    diameter=10,
    sigma_color=75,
    sigma_space=75,
    clip_limit=2.0,
    title_grid_size=(8, 8),
) -> np.ndarray:
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize, bilateral filter, and CLAHE
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = cv2.bilateralFilter(image, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    image = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=title_grid_size).apply(image)

    return image


def fingertip_thresholding(image: np.ndarray, blockSize=11) -> np.ndarray:
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, 2
    )


"""
Fingerprint Processing Functions
"""
_sigma_conv = (3.0 / 2.0) / ((6 * math.log(10)) ** 0.5)


def _gabor_sigma(ridge_period: float):
    return _sigma_conv * ridge_period


def _gabor_size(ridge_period: float):
    """
    Calculate the size of the Gabor filter based on the ridge period.

    The size is calculated as twice the ridge period plus one, rounded to the nearest odd integer.
    This is to ensure that the filter size is always odd, which is a common requirement for convolutional filters.

    Args:
        ridge_period (float): The ridge period of the fingerprint image, typically measured in pixels.

    Returns:
        tuple: A tuple of two integers representing the size of the Gabor filter (height, width).

    Reference:
        This function is based on the implementation of
        "Hands on Fingerprint Recognition with OpenCV and Python" by Raffaele Cappelli.
        https://tinyurl.com/hands-on-fr
    """
    p = int(round(ridge_period * 2 + 1))

    if p % 2 == 0:
        p += 1

    return (p, p)


def _gabor_kernel(period: float, orientation: float) -> np.ndarray:
    """
    Generate a Gabor filter kernel.

    The Gabor filter is generated using the OpenCV function `cv2.getGaborKernel`.
    The size and sigma of the filter are determined based on the ridge period of the fingerprint image.
    The orientation of the filter is specified by the `orientation` argument.

    Args:
        period (float): The ridge period of the fingerprint image, typically measured in pixels.
        orientation (float): The orientation of the ridges in the fingerprint image, in radians.

    Returns:
        ndarray: A 2D array representing the Gabor filter kernel.

    Reference:
        This function is based on the implementation of
        "Hands on Fingerprint Recognition with OpenCV and Python" by Raffaele Cappelli.
        https://tinyurl.com/hands-on-fr
    """
    f = cv2.getGaborKernel(
        _gabor_size(period),
        _gabor_sigma(period),
        np.pi / 2 - orientation,
        period,
        gamma=1,
        psi=0,
    )
    f /= f.sum()
    f -= f.mean()

    return f


def _to_fingerprint(image: np.array, width: int = 400) -> np.array:
    """
    Convert an image to a fingerprint image.

    This function applies several image processing steps to enhance the fingerprint ridges and suppress the noise.
    These steps include resizing the image, calculating the local gradient, estimating the ridge line period,
    creating a Gabor filter bank, filtering the image with each filter in the bank, and finally converting the
    result to grayscale.

    Args:
        image (np.array): The input image, assumed to be a grayscale fingerprint image.
        width (int, optional): The width to which the input image is resized. Defaults to 400.

    Returns:
        np.array: The enhanced fingerprint image.

    Reference:
        This function is based on the implementation of
        "Hands on Fingerprint Recognition with OpenCV and Python" by Raffaele Cappelli.
        https://tinyurl.com/hands-on-fr
    """
    fingerprint = imutils.resize(image.copy(), width)

    # Calculate the local gradient (using Sobel filters)
    gx, gy = cv2.Sobel(fingerprint, cv2.CV_32F, 1, 0), cv2.Sobel(fingerprint, cv2.CV_32F, 0, 1)

    # Calculate the magnitude of the gradient for each pixel
    gx2, gy2 = gx**2, gy**2

    window = (29, 29)
    gxx = cv2.boxFilter(gx2, -1, window, normalize=False)
    gyy = cv2.boxFilter(gy2, -1, window, normalize=False)
    gxy = cv2.boxFilter(gx * gy, -1, window, normalize=False)
    gxx_gyy = gxx - gyy
    gxy2 = 2 * gxy

    orientations = (cv2.phase(gxx_gyy, -gxy2) + np.pi) / 2

    center_h = fingerprint.shape[0] // 3
    center_w = fingerprint.shape[1] // 2
    region = fingerprint[center_h - 40 : center_h + 40, center_w + 10 : center_w + 60]

    # before computing the x-signature, the region is smoothed to reduce noise
    smoothed = cv2.blur(region, (5, 5), -1)
    xs = np.sum(smoothed, 1)  # the x-signature of the region

    # Find the indices of the x-signature local maxima
    local_maxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]

    # Calculate all the distances between consecutive peaks
    distances = local_maxima[1:] - local_maxima[:-1]

    # Estimate the ridge line period as the average of the above distances
    ridge_period = np.average(distances)

    # Create the filter bank
    or_count = 8
    gabor_bank = [_gabor_kernel(ridge_period, o) for o in np.arange(0, np.pi, np.pi / or_count)]

    # Filter the whole image with each filter
    nf = 255 - fingerprint
    all_filtered = np.array([cv2.filter2D(nf, cv2.CV_32F, f) for f in gabor_bank])

    y_coords, x_coords = np.indices(fingerprint.shape)

    # For each pixel, find the index of the closest orientation in the gabor bank
    orientation_idx = (
        np.round(((orientations % np.pi) / np.pi) * or_count).astype(np.int32) % or_count
    )
    # Take the corresponding convolution result for each pixel, to assemble the final result
    filtered = all_filtered[orientation_idx, y_coords, x_coords]

    # Convert to gray scale and apply the mask
    gray = np.clip(filtered, 0, 255).astype(np.uint8)

    return gray


@memory.cache
def fingerprint_mapping(image: np.ndarray) -> np.ndarray:
    try:
        fingerprint = _to_fingerprint(image)

        return fingerprint
    except Exception as e:
        print(f"Error converting fingerphoto to fingerprint: {e}")
        return None


@memory.cache
def fingerprint_enhancement(image: np.ndarray) -> np.ndarray:
    """
    Enhance a fingerprint image.

    This function uses a fingerprint enhancer to improve the clarity of the fingerprint ridges.
    The enhancer is assumed to be an instance of a FingerprintEnhancer class, which has an `enhance_Fingerprint` method.

    The enhancement process includes the following steps:

        1. Normalise the image and find a Region of Interest (ROI).
        2. Compute the orientation image.
        3. Compute the major frequency of the ridges.
        4. Filter the image using an oriented Gabor filter.

    Args:
        image (np.ndarray): The input image, assumed to be a grayscale fingerprint image.

    Returns:
        np.ndarray: The enhanced fingerprint image.

    Reference:
        This function used the implementation of "Fingerprint-Enhancement-Python" by Utkarsh Deshmukh.
        https://github.com/Utkarsh-Deshmukh/Fingerprint-Enhancement-Python
    """
    fingerprint = fpe.enhance_Fingerprint(image)
    fingerprint = fingerprint.astype("uint8")

    return fingerprint
