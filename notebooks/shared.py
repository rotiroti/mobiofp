import warnings

# Suppress Pandas 3.0 PyArrow warning
warnings.filterwarnings("ignore")

import math
from pathlib import Path

import cv2
import imutils
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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


def show_distributions(df: pd.DataFrame) -> None:
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(df.columns[1:]):
        plt.subplot(1, 3, i + 1)
        sns.kdeplot(df[col], shade=True, color="skyblue", alpha=0.6)
        plt.title(f"{col} Histogram")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.axvline(df[col].mean(), color="r", linestyle="--", label=f"Mean: {df[col].mean():.2f}")
        plt.axvline(
            df[col].median(), color="g", linestyle="--", label=f"Median: {df[col].median():.2f}"
        )
        plt.legend()
    plt.tight_layout()
    plt.show()


def show_boxplots(df: pd.DataFrame) -> None:
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(df.columns[1:]):
        plt.subplot(1, 3, i + 1)
        sns.boxplot(y=col, data=df, color="skyblue")
        plt.title(f"{col} Boxplot")
        plt.ylabel(col)
    plt.tight_layout()
    plt.show()


def data_loss(original_df: pd.DataFrame, filtered_df: pd.DataFrame) -> pd.DataFrame:
    original_count = len(original_df)
    filtered_count = len(filtered_df)
    data_loss = original_count - filtered_count
    percentage_loss = (data_loss / original_count) * 100

    print(f"Original dataset size: {original_count} images")
    print(f"Filtered dataset size: {filtered_count} images")
    print(f"Data loss: {data_loss} images/masks ({percentage_loss:.2f}% loss)")

    # Extract indices of data loss images
    data_loss_indices = original_df[~original_df.index.isin(filtered_df.index)]

    # Return DataFrame containing only data loss images
    data_loss_df = original_df.loc[data_loss_indices.index]

    return data_loss_df
