import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional

def plot_image(image: np.array, title: Optional[str] = "Original Image") -> None:
    """
    Plots a single image.

    Parameters:
        image (np.array): The image to be plotted.
        title (str): The title of the plot.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis("off")
    plt.show()

def plot_images(images: [np.array], titles: [str], rows: int = 1, cols: int = 1) -> None:
    """
    Plots a list of images.

    Parameters:
        images (List[np.array]): The images to be plotted.
        titles (List[str]): The titles of the plots.
        subplot_shape (Tuple[int, int]): The shape of the subplot (nrows, ncols).
    """
    assert len(images) == len(titles), "The number of images and titles must be the same."

    plt.figure(figsize=(10, 10))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i+1)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

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
    plt.imshow(image, cmap='gray')
    plt.title(title)

    # Plot the histogram with CDF
    plt.subplot(122)
    plt.plot(cdf_normalized, color='b')
    plt.hist(image.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.title('Histogram')

    plt.tight_layout()
    plt.show()

def plot_img_bbox(image: np.array, bbox: tuple, image_title: Optional[str] = "Original Image", bbox_title: Optional[str] = "Class Name") -> None:
    """
    Plots the image with a bounding box and a title.

    Parameters:
        image (np.array): The image to be plotted.
        bbox (tuple): The bounding box to be drawn on the image.
    """
    bbox_image = cv2.rectangle(image.copy(), bbox, (255,0,0), 5)
    cv2.putText(bbox_image, bbox_title, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,0,0), 5)

    plt.figure(figsize=(12, 18))
    plt.imshow(bbox_image)
    plt.title(image_title)
    plt.axis("off")
    plt.show()
