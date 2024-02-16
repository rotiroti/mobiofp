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


def _find_largest_connected_component(mask: np.ndarray) -> np.array:
    """
    Finds the largest connected component in a binary mask.
    Args:
        mask: Binary mask containing connected components.
    Returns:
        Binary mask with only the largest connected component.
    """
    # Find connected components in the mask
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Find the label of the largest connected component (excluding the background)
    largest_component_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # Create a binary mask where the largest connected component is white and everything else is black
    largest_component_mask = np.where(labels == largest_component_label, 255, 0).astype(np.uint8)

    return largest_component_mask


def extract_roi(mask: np.ndarray, factor: float = 1.0) -> tuple[int, int, int, int]:
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


def crop_image(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
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


def post_process_mask(mask: np.ndarray) -> np.ndarray:
    # Apply morphological operation and Gaussian blur
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
    mask = np.where(mask < 127, 0, 255).astype(np.uint8)

    # Find the largest connected component
    mask = _find_largest_connected_component(mask)

    return mask


def sharpness_score(image: np.ndarray) -> float:
    """
    Calculates the sharpness score of an image using the Laplacian method.

    Parameters:
        image (np.ndarray): Input image.

    Returns:
        float: Sharpness score of the image.
    """
    laplacian = cv2.Laplacian(image.copy(), cv2.CV_64F)
    score = laplacian.var()

    return score


def contrast_score(image: np.ndarray) -> float:
    """
    Calculates the contrast score of an image using histogram analysis.

    Parameters:
        image (np.ndarray): Input image.

    Returns:
        float: Contrast score of the image.
    """
    hist, _ = np.histogram(image, bins=256, range=[0, 256])
    cumulative_hist = np.cumsum(hist)
    total_pixels = image.shape[0] * image.shape[1]
    low_threshold = 0.05 * total_pixels
    high_threshold = 0.95 * total_pixels
    low_intensity = np.argmax(cumulative_hist > low_threshold)
    high_intensity = np.argmax(cumulative_hist > high_threshold)
    score = high_intensity - low_intensity

    return score


def coverage_percentage(mask: np.ndarray) -> float:
    """
    Calculates the percentage of the image covered by a mask.

    Parameters:
        mask (np.ndarray): Input mask.

    Returns:
        float: Coverage percentage.
    """
    total_pixels = np.sum(mask > 0)
    percentage = (total_pixels / (mask.shape[0] * mask.shape[1])) * 100

    return percentage


def quality_scores(image: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """
    Computes the quality scores (sharpness, contrast, coverage) of an image.

    Parameters:
        image (np.ndarray): Input image.
        mask (np.ndarray): Input mask.

    Returns:
        tuple: A tuple containing the sharpness, contrast, and coverage scores.
    """
    sharpness = sharpness_score(image)
    contrast = contrast_score(image)
    coverage = coverage_percentage(mask)

    return sharpness, contrast, coverage


def fingertip_enhancement(image):
    # Convert to grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize, bilateral filter, and CLAHE
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = cv2.bilateralFilter(image, 7, 50, 50)
    image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)

    return image


"""
Fingerprint Processing Functions
"""
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
        This function is based on the implementation of
        "Hands on Fingerprint Recognition with OpenCV and Python" by Raffaele Cappelli.
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
    # Fingertip Adaptive Thresholding
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2)

    try:
        fingerprint = _to_fingerprint(binary)

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


"""
Feauture Extraction and Matching Functions
"""


def imkpts(image, kpts, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS):
    result = cv2.drawKeypoints(image, kpts, None, flags=flags)

    return result


def orb_bf_matcher(img1, kp1, desc1, img2, kp2, desc2, include_img=False):
    # Create a brute-force matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Perform the matching between the two descriptor sets
    matches = bf.match(desc1, desc2)

    # Sort the matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Compute median distance
    distance = np.median([m.distance for m in matches])

    if not include_img:
        return distance

    # Draw the matches
    matches_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)

    return distance, matches_img


def orb_flann_matcher(img1, kp1, desc1, img2, kp2, desc2, include_img=False):
    index_params = {
        "algorithm": 6,  # FLANN_INDEX_LSH,
        "table_number": 6,
        "key_size": 12,
        "multi_probe_level": 1,
    }
    search_params = {"checks": 50}

    # Create a FLANN matcher object
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    matchesMask = [[0, 0] for i in range(len(matches))]
    good_matches = []

    for i, match in enumerate(matches):
        if len(match) == 2:
            m, n = match
            if m.distance < 0.75 * n.distance:
                matchesMask[i] = [1, 0]
                good_matches.append(m)

    # Compute median distance
    if good_matches:
        distance = np.median([m.distance for m in good_matches])
        # Check if distance is nan or inf, if so, set it to a large number
        if np.isnan(distance) or np.isinf(distance):
            distance = float("inf")
    else:
        distance = float("inf")

    if not include_img:
        return distance

    draw_params = {
        "matchesMask": matchesMask,
        "flags": cv2.DrawMatchesFlags_DEFAULT,
    }

    # Draw the good matches
    matches_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    return distance, matches_img
