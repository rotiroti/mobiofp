import cv2
import math
import imutils
import numpy as np
import fingerprint_feature_extractor

from typing import Optional

def find_largest_connected_component(mask: np.ndarray) -> np.array:
    """
    Finds the largest connected component in a binary mask.

    Args:
        mask: Binary mask containing connected components.
    Returns:
        Binary mask with only the largest connected component.
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Find the index of the largest connected component (excluding background)
    largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # Create a mask with only the largest connected component
    largest_component_mask = np.uint8(labels == largest_component_index)

    return largest_component_mask

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

def calculate_orientation_angle(contour_points: np.ndarray, image: np.ndarray) -> float:
    """
    Calculate the orientation angle of a set of contour points in an image.

    This function uses Principal Component Analysis (PCA) to calculate the
    orientation angle of a set of contour points. The orientation angle is the angle
    between the first principal component and the horizontal axis.

    Args:
        contour_points (np.ndarray): A numpy array of shape (N, 2) where N is the number of points. Each row is a point (x, y).
        image (np.ndarray): The image in which the contour points are located.

    Returns:
        float: The orientation angle in radians.
    """
    num_points = len(contour_points)
    data_points = np.empty((num_points, 2), dtype=np.float64)
    
    for i in range(num_points):
        data_points[i,0] = contour_points[i,0,0]
        data_points[i,1] = contour_points[i,0,1]
    
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, _ = cv2.PCACompute2(data_points, mean)
    
    # Calculate orientation in radians
    orientation_angle = math.atan2(eigenvectors[0,1], eigenvectors[0,0])

    return orientation_angle

def fix_orientation(image: np.ndarray, mask: np.ndarray) -> (np.ndarray, np.ndarray, int):
    """
    Fix the orientation of an image and its mask.

    Args:
        image (np.ndarray): The image to be fixed.
        mask (np.ndarray): The mask to be fixed.

    Returns:
        tuple[np.ndarray, np.ndarray, int]: A tuple containing the fixed image, the fixed mask and the orientation angle in degrees.
    """
    # Find the largest connected component in the mask
    lcc = find_largest_connected_component(mask)

    # Find the contour points of the largest connected component
    contours, _ = cv2.findContours(lcc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_points = contours[0]

    # Calculate the orientation angle
    angle = calculate_orientation_angle(contour_points, image)

    # Convert the angle to degrees
    angle = math.degrees(angle)

    image = imutils.rotate_bound(image, angle)
    mask = imutils.rotate_bound(mask, angle)

    return image, mask, angle

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate an image by a given angle.

    Args:
        image (np.ndarray): The image to be rotated.
        angle (float): The angle in degrees.
    
    Returns:
        np.ndarray: The rotated image.
    """
    return imutils.rotate_bound(image, angle)

def save_images(image: np.array, image_path: str, mask: Optional[np.array] = None, mask_path: Optional[str] = None) -> None:
    """
    Saves the image and an optional mask.

    Args:
        image (np.array): The image to be saved.
        image_path (str): The path where the image will be saved.
        mask (np.array, optional): The mask to be saved. If None, the mask is not saved. Defaults to None.
        mask_path (str, optional): The path where the mask will be saved. If None, the mask is not saved. Defaults to None.
    """
    print(f"Saving image at: {image_path}")
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(image_path, image)

    if mask is not None and mask_path is not None:
        print(f"Saving mask at: {mask_path}")
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(mask_path, mask)

_sigma_conv = (3.0/2.0)/((6*math.log(10))**0.5)

def _gabor_sigma(ridge_period):
    return _sigma_conv * ridge_period

def _gabor_size(ridge_period):
    p = int(round(ridge_period * 2 + 1))

    if p % 2 == 0:
        p += 1

    return (p, p)

def _gabor_kernel(period, orientation):
    f = cv2.getGaborKernel(_gabor_size(period), _gabor_sigma(period), np.pi/2 - orientation, period, gamma = 1, psi = 0)
    f /= f.sum()
    f -= f.mean()
    
    return f

def to_fingerprint(image: np.array, width: int = 400) -> np.array:
    fingerprint = image.copy()
    fingerprint = imutils.resize(fingerprint, width)

    # Calculate the local gradient (using Sobel filters)
    gx, gy = cv2.Sobel(fingerprint, cv2.CV_32F, 1, 0), cv2.Sobel(fingerprint, cv2.CV_32F, 0, 1)

    # Calculate the magnitude of the gradient for each pixel
    gx2, gy2 = gx**2, gy**2

    W = (29, 29) # (23, 23)
    gxx = cv2.boxFilter(gx2, -1, W, normalize = False)
    gyy = cv2.boxFilter(gy2, -1, W, normalize = False)
    gxy = cv2.boxFilter(gx * gy, -1, W, normalize = False)
    gxx_gyy = gxx - gyy
    gxy2 = 2 * gxy

    orientations = (cv2.phase(gxx_gyy, -gxy2) + np.pi) / 2 # '-' to adjust for y axis direction

    # _region = fingerprint[10:90,80:130]
    center_h = fingerprint.shape[0]//3
    center_w = fingerprint.shape[1]//2
    #_region = fingerprint[center_h-40:center_h+40, center_w-25:center_w+25]
    region = fingerprint[center_h-40:center_h+40, center_w+10:center_w+60]

    # before computing the x-signature, the region is smoothed to reduce noise
    smoothed = cv2.blur(region, (5,5), -1)
    xs = np.sum(smoothed, 1) # the x-signature of the region

    # Find the indices of the x-signature local maxima
    local_maxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]

    # Calculate all the distances between consecutive peaks
    distances = local_maxima[1:] - local_maxima[:-1]

    # Estimate the ridge line period as the average of the above distances
    ridge_period = np.average(distances)

    # Create the filter bank
    or_count = 8
    gabor_bank = [_gabor_kernel(ridge_period, o) for o in np.arange(0, np.pi, np.pi/or_count)]

    # Filter the whole image with each filter
    # Note that the negative image is actually used, to have white ridges on a black background as a result!!
    nf = 255-fingerprint
    all_filtered = np.array([cv2.filter2D(nf, cv2.CV_32F, f) for f in gabor_bank])

    y_coords, x_coords = np.indices(fingerprint.shape)
    # For each pixel, find the index of the closest orientation in the gabor bank
    orientation_idx = np.round(((orientations % np.pi) / np.pi) * or_count).astype(np.int32) % or_count
    # Take the corresponding convolution result for each pixel, to assemble the final result
    filtered = all_filtered[orientation_idx, y_coords, x_coords]
    # Convert to gray scale and apply the mask
    enhanced = np.clip(filtered, 0, 255).astype(np.uint8)

    return enhanced

def extract_minutiae(fingerprint: np.ndarray, threshold: int = 10, invert: bool = False, save: bool = False) -> (np.ndarray, np.ndarray):
    terminations, bifurcations = fingerprint_feature_extractor.extract_minutiae_features(fingerprint, spuriousMinutiaeThresh=threshold, invertImage=invert, saveResult=save)

    return terminations, bifurcations

# class GaborFilter:
#     _sigma_conv = (3.0/2.0)/((6*math.log(10))**0.5)

#     def _sigma(self, ridge_period):
#         return self._sigma_conv * ridge_period

#     def _size(self, ridge_period):
#         p = int(round(ridge_period * 2 + 1))

#         if p % 2 == 0:
#             p += 1

#         return (p, p)

#     def _create_kernel(self, period, orientation):
#         f = cv2.getGaborKernel(self._size(period), self._sigma(period), np.pi/2 - orientation, period, gamma = 1, psi = 0)
#         f /= f.sum()
#         f -= f.mean()

#         return f

#     def orientation(self, image):
#         fingerprint = image.copy()

#         imutils.resize(fingerprint, width=400)

#         # Calculate the local gradient (using Sobel filters)
#         gx, gy = cv2.Sobel(fingerprint, cv2.CV_32F, 1, 0), cv2.Sobel(fingerprint, cv2.CV_32F, 0, 1)

#         # Calculate the magnitude of the gradient for each pixel
#         gx2, gy2 = gx**2, gy**2

#         W = (29, 29) # (23, 23)
#         gxx = cv2.boxFilter(gx2, -1, W, normalize = False)
#         gyy = cv2.boxFilter(gy2, -1, W, normalize = False)
#         gxy = cv2.boxFilter(gx * gy, -1, W, normalize = False)
#         gxx_gyy = gxx - gyy
#         gxy2 = 2 * gxy

#         orientations = (cv2.phase(gxx_gyy, -gxy2) + np.pi) / 2 # '-' to adjust for y axis direction

#         center_h = fingerprint.shape[0]//3
#         center_w = fingerprint.shape[1]//2
#         region = fingerprint[center_h-40:center_h+40, center_w+10:center_w+60]

#         # before computing the x-signature, the region is smoothed to reduce noise
#         smoothed = cv2.blur(region, (5,5), -1)
#         xs = np.sum(smoothed, 1) # the x-signature of the region

#         # Find the indices of the x-signature local maxima
#         local_maxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]

#         # Calculate all the distances between consecutive peaks
#         distances = local_maxima[1:] - local_maxima[:-1]

#         # Estimate the ridge line period as the average of the above distances
#         ridge_period = np.average(distances)

#         # Create the filter bank
#         or_count = 8
#         gabor_bank = [self._create_kernel(ridge_period, o) for o in np.arange(0, np.pi, np.pi/or_count)]

#         # Filter the whole image with each filter
#         # Note that the negative image is actually used, to have white ridges on a black background as a result!!
#         nf = 255-fingerprint
#         all_filtered = np.array([cv2.filter2D(nf, cv2.CV_32F, f) for f in gabor_bank])

#         y_coords, x_coords = np.indices(fingerprint.shape)

#         # For each pixel, find the index of the closest orientation in the gabor bank
#         orientation_idx = np.round(((orientations % np.pi) / np.pi) * or_count).astype(np.int32) % or_count

#         # Take the corresponding convolution result for each pixel, to assemble the final result
#         filtered = all_filtered[orientation_idx, y_coords, x_coords]

#         # Convert to gray scale and apply the mask
#         enhanced = np.clip(filtered, 0, 255).astype(np.uint8)

#         return enhanced

# _sigma_conv = (3.0/2.0)/((6*math.log(10))**0.5)
# # sigma is adjusted according to the ridge period, so that the filter does not contain more than three effective peaks
# def _gabor_sigma(ridge_period):
#     return _sigma_conv * ridge_period

# def _gabor_size(ridge_period):
#     p = int(round(ridge_period * 2 + 1))
#     if p % 2 == 0:
#         p += 1
#     return (p, p)

# def gabor_kernel(period, orientation):
#     f = cv2.getGaborKernel(_gabor_size(period), _gabor_sigma(period), np.pi/2 - orientation, period, gamma = 1, psi = 0)
#     f /= f.sum()
#     f -= f.mean()
#     return f
