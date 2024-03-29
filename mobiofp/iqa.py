import cv2
import numpy as np
from imutils import auto_canny
from skimage.restoration import estimate_sigma


def estimate_noise(image, channel_axis=-1) -> float:
    return estimate_sigma(image, average_sigmas=True, channel_axis=-1)


def rms_contrast(image) -> float:
    return np.sqrt(np.mean(np.square(image - np.mean(image))))


def canny_sharpness(image, auto=True, th_min=100, th_max=200) -> float:
    if auto:
        return np.sum(auto_canny(image) > 0) / (image.shape[0] * image.shape[1])

    return np.sum(cv2.Canny(image, th_min, th_max) > 0) / (image.shape[0] * image.shape[1])


def edge_density(image, square_size=50) -> float:
    # Calculate the coordinates of the center of the image
    center_y, center_x = image.shape[0] // 2, image.shape[1] // 2

    # Calculate the coordinates of the top left and bottom right corners of the square
    top_left_y = max(0, center_y - square_size // 2)
    top_left_x = max(0, center_x - square_size // 2)
    bottom_right_y = min(image.shape[0], center_y + square_size // 2)
    bottom_right_x = min(image.shape[1], center_x + square_size // 2)

    # Extract the square from the image
    square = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # Compute the edge density on the square
    edge_image = gradient_magnitude(square)

    return np.sum(edge_image > 0) / (square.shape[0] * square.shape[1])


def laplacian_sharpness(image) -> float:
    return cv2.Laplacian(image, cv2.CV_64F).var()


def gradient_magnitude(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    return np.sqrt(gx**2 + gy**2)
