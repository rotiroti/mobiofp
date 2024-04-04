import math
from pathlib import Path

import cv2
import imutils
import matplotlib.pyplot as plt
from tqdm import tqdm


def read_images(
    images_directory,
    file_extesion="jpg",
    grayscale=False,
    resize=False,
    resize_width=640,
    rotate=False,
    rotate_angle=0,
):
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


def save_images(images, titles, target_directory, file_extesion="jpg", mask=False):
    """
    Save images to a directory.
    """
    target_directory = Path(target_directory)
    target_directory.mkdir(parents=True, exist_ok=True)

    for img, title in zip(images, titles):
        img_path = target_directory / f"{title}.{file_extesion}"

        if not mask:
            cv2.imwrite(str(img_path), img)
        else:
            cv2.imwrite(str(img_path), img * 255)

    print(f"Images saved to {target_directory}")


def show_images(
    images, titles, cmap="gray", show_axis=False, fig_size=10, sup_title=None, limit=10
):
    assert (titles is None) or (len(images) == len(titles))

    # Limit the number of images
    images = images[:limit]
    if titles is not None:
        titles = titles[:limit]

    num_images = len(images)
    num_cols = 4
    num_rows = math.ceil(num_images / num_cols)

    fig_height = fig_size * (num_rows / num_cols) * 1.5
    _, axes = plt.subplots(
        num_rows, num_cols, figsize=(fig_size, fig_height), constrained_layout=True
    )
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
