import cv2
import math
import imutils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from tqdm import tqdm
from joblib import Memory
from mobiofp import to_fingerprint, enhance_fingerprint, find_largest_connected_component

memory = Memory("cache", verbose=0)

def show_images(images, titles, cmap="gray", show_axis=False, fig_size=10, sup_title=None):
    assert((titles is None) or (len(images) == len(titles)))

    num_images = len(images)
    num_cols = 4
    num_rows = math.ceil(num_images / num_cols)

    fig_height = fig_size * (num_rows / num_cols) * 1.5
    _, axes = plt.subplots(num_rows, num_cols, figsize=(fig_size, fig_height), constrained_layout=True)
    axes = axes.ravel()

    if sup_title:
        plt.suptitle(sup_title, fontsize=24)

    for idx, (image, title) in enumerate(zip(images, titles)):
        axes[idx].imshow(image, cmap=cmap)
        axes[idx].set_title(title, fontsize=12)
        axes[idx].axis("on" if show_axis else "off")
    
    # Hide the remaining subplots
    for idx in range(num_images, num_cols * num_rows):
        axes[idx].axis("off")

plt.show()

def show_iqa(sharpness_scores, contrast_scores, mask_coverage_scores):
    _, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
    plt.suptitle("Fingertip Image-Quality Assesment", fontsize=16)

    sns.boxplot(sharpness_scores, ax=axes[0], color="blue")
    axes[0].set_xlabel("Sharpness Score")
    axes[0].set_ylabel("Density")

    sns.boxplot(contrast_scores, ax=axes[1], color="red")
    axes[1].set_xlabel("Contrast Score")
    axes[1].set_ylabel("Density")

    sns.boxplot(mask_coverage_scores, ax=axes[2], color="green")
    axes[2].set_xlabel("Binary Mask Coverage Score")
    axes[2].set_ylabel("Density")

    plt.show()

def post_process_mask(mask):
    # Apply morphological operation and Gaussian blur
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
    mask = np.where(mask < 127, 0, 255).astype(np.uint8)

    # Find the largest connected component
    mask = find_largest_connected_component(mask)

    return mask

def fingertip_enhancement(image):
    # Convert to grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize, bilateral filter, and CLAHE
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = cv2.bilateralFilter(image, 7, 50, 50)
    image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)

    return image

@memory.cache
def from_fingertip_to_fingerprint(image):
    # Fingertip Adaptive Thresholding
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2)

    try:
        fingerprint = to_fingerprint(binary)

        return fingerprint
    except Exception as e:
        print(f"Error converting fingerphoto to fingerprint: {e}")
        return None

@memory.cache
def fingerprint_enhancement(image):
    fingerprint = enhance_fingerprint(image)
    fingerprint = fingerprint.astype("uint8")

    return fingerprint

@memory.cache
def read_images(images_directory, file_extesion="jpg", grayscale=False, resize=False, resize_width=640, rotate=False, rotate_angle=0):
    image_paths = list(Path(images_directory).glob(f"*.{file_extesion}"))
    images = []
    images_titles = []

    for p in tqdm(image_paths):
        img = cv2.imread(str(p))

        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if resize:
            img = imutils.resize(img, width=resize_width)

        if rotate:
            img = imutils.rotate_bound(img, rotate_angle)

        images.append(img)
        images_titles.append(p.stem)
    
    return images, images_titles
