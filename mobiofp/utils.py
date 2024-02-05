"""
This file contains the implementation of some utility functions used in the project.
"""

import math
from typing import Optional

import cv2
import fingerprint_enhancer as fpe
import fingerprint_feature_extractor as ffe
import imutils
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize


def plot_img_hist(image: np.array, title: Optional[str] = "Original Image") -> None:
    """
    Plots the image and its histogram with CDF.

    Parameters:
        image (np.array): The image to be plotted.
    """
    # Calculate histogram and CDF
    hist, _ = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Create the plots
    plt.figure(figsize=(15, 5))

    # Plot the image
    plt.subplot(121)
    plt.imshow(image, cmap="gray")
    plt.title(title)

    # Plot the histogram with CDF
    plt.subplot(122)
    plt.plot(cdf_normalized, color="b")
    plt.hist(image.flatten(), 256, [0, 256], color="r")
    plt.xlim([0, 256])
    plt.legend(("cdf", "histogram"), loc="upper left")
    plt.title("Histogram")

    plt.tight_layout()
    plt.show()


def extract_roi(mask: np.ndarray, factor: float = 1.10) -> (int, int, int, int):
    """
    Extract ROI from a binary mask.

    Args:
        mask: Binary mask.
        factor: Factor to increase the size of the ROI.
    Returns:
        Tuple with four coordinates representing the bounding box rectangle.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    # Adjust the size of the ROI by a factor
    w_new = int(w * factor)
    h_new = int(h * factor)
    x_new = max(0, x - (w_new - w) // 2)
    y_new = max(0, y - (h_new - h) // 2)

    return (x_new, y_new, w_new, h_new)


def crop_image(image: np.ndarray, roi: (int, int, int, int)) -> np.ndarray:
    """
    Crop an image or a mask using a bounding box.

    Args:
        image: Image to be cropped.
        roi: Tuple with four coordinates representing the bounding box rectangle.
    Returns:
        Cropped image.
    """
    x, y, w, h = roi

    return image[y : y + h, x : x + w]


_sigma_conv = (3.0 / 2.0) / ((6 * math.log(10)) ** 0.5)


def _gabor_sigma(ridge_period):
    return _sigma_conv * ridge_period


def _gabor_size(ridge_period):
    """
    Calculate the size of the Gabor filter based on the ridge period.

    The size is calculated as twice the ridge period plus one, rounded to the nearest odd integer.
    This is to ensure that the filter size is always odd, which is a common requirement for convolutional filters.

    Args:
        ridge_period (float): The ridge period of the fingerprint image, typically measured in pixels.

    Returns:
        tuple: A tuple of two integers representing the size of the Gabor filter (height, width).

    Reference:
        This function is based on the implementation of "Hands on Fingerprint Recognition with OpenCV and Python" by Raffaele Cappelli.
        https://tinyurl.com/hands-on-fr
    """
    p = int(round(ridge_period * 2 + 1))

    if p % 2 == 0:
        p += 1

    return (p, p)


def _gabor_kernel(period, orientation):
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
        This function is based on the implementation of "Hands on Fingerprint Recognition with OpenCV and Python" by Raffaele Cappelli.
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


def to_fingerprint(image: np.array, width: int = 400) -> np.array:
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
        This function is based on the implementation of "Hands on Fingerprint Recognition with OpenCV and Python" by Raffaele Cappelli.
        https://tinyurl.com/hands-on-fr
    """
    fingerprint = imutils.resize(image.copy(), width)

    # Calculate the local gradient (using Sobel filters)
    gx, gy = cv2.Sobel(fingerprint, cv2.CV_32F, 1, 0), cv2.Sobel(
        fingerprint, cv2.CV_32F, 0, 1
    )

    # Calculate the magnitude of the gradient for each pixel
    gx2, gy2 = gx**2, gy**2

    W = (29, 29)
    gxx = cv2.boxFilter(gx2, -1, W, normalize=False)
    gyy = cv2.boxFilter(gy2, -1, W, normalize=False)
    gxy = cv2.boxFilter(gx * gy, -1, W, normalize=False)
    gxx_gyy = gxx - gyy
    gxy2 = 2 * gxy

    orientations = (cv2.phase(gxx_gyy, -gxy2) + np.pi) / 2

    # _region = fingerprint[10:90,80:130]
    center_h = fingerprint.shape[0] // 3
    center_w = fingerprint.shape[1] // 2
    region = fingerprint[center_h - 40 : center_h + 40, center_w + 10 : center_w + 60]

    # before computing the x-signature, the region is smoothed to reduce noise
    smoothed = cv2.blur(region, (5, 5), -1)
    xs = np.sum(smoothed, 1)  # the x-signature of the region

    # Find the indices of the x-signature local maxima
    local_maxima = np.nonzero(
        np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False]
    )[0]

    # Calculate all the distances between consecutive peaks
    distances = local_maxima[1:] - local_maxima[:-1]

    # Estimate the ridge line period as the average of the above distances
    ridge_period = np.average(distances)

    # Create the filter bank
    or_count = 8
    gabor_bank = [
        _gabor_kernel(ridge_period, o) for o in np.arange(0, np.pi, np.pi / or_count)
    ]

    # Filter the whole image with each filter
    nf = 255 - fingerprint
    all_filtered = np.array([cv2.filter2D(nf, cv2.CV_32F, f) for f in gabor_bank])

    y_coords, x_coords = np.indices(fingerprint.shape)

    # For each pixel, find the index of the closest orientation in the gabor bank
    orientation_idx = (
        np.round(((orientations % np.pi) / np.pi) * or_count).astype(np.int32)
        % or_count
    )
    # Take the corresponding convolution result for each pixel, to assemble the final result
    filtered = all_filtered[orientation_idx, y_coords, x_coords]

    # Convert to gray scale and apply the mask
    enhanced = np.clip(filtered, 0, 255).astype(np.uint8)

    return enhanced


def enhance_fingerprint(image: np.ndarray) -> np.ndarray:
    """
    Enhance a fingerprint image.

    This function uses a fingerprint enhancer to improve the clarity of the fingerprint ridges.
    The enhancer is assumed to be an instance of a FingerprintEnhancer class, which has an `enhance_Fingerprint` method.

    Args:
        image (np.ndarray): The input image, assumed to be a grayscale fingerprint image.

    Returns:
        np.ndarray: The enhanced fingerprint image.

    Reference:
        This function used the implementation of "Fingerprint-Enhancement-Python" by Utkarsh Deshmukh.
        https://github.com/Utkarsh-Deshmukh/Fingerprint-Enhancement-Python
    """
    return fpe.enhance_Fingerprint(image)


def extract_minutiae(image: np.ndarray, thresh: int = 25):
    terminations, bifurcations = ffe.extract_minutiae_features(image, thresh)

    print(f"Total minutiae: {len(terminations) + len(bifurcations)}")
    print(f"{len(terminations)} terminations.")
    print(f"{len(bifurcations)} bifurcations.")

    return terminations, bifurcations


def skeleton(image: np.ndarray) -> np.ndarray:
    return skeletonize(image).astype(np.uint8) * 255


def show_minutiae(image: np.ndarray, minutiae: list):
    pass

    # def showResults(self, FeaturesTerm, FeaturesBif):

    #     (rows, cols) = self._skel.shape
    #     DispImg = np.zeros((rows, cols, 3), np.uint8)
    #     DispImg[:, :, 0] = 255*self._skel
    #     DispImg[:, :, 1] = 255*self._skel
    #     DispImg[:, :, 2] = 255*self._skel

    #     for idx, curr_minutiae in enumerate(FeaturesTerm):
    #         row, col = curr_minutiae.locX, curr_minutiae.locY
    #         (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
    #         skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))

    #     for idx, curr_minutiae in enumerate(FeaturesBif):
    #         row, col = curr_minutiae.locX, curr_minutiae.locY
    #         (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
    #         skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0))

    #     cv2.imshow('output', DispImg)
    #     cv2.waitKey(0)
