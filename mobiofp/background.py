import cv2
import numpy as np
from rembg import new_session, remove


class BackgroundRemoval:
    def __init__(self, session="u2net"):
        self.session = new_session(session)

    def apply(self, image):
        mask = remove(image, session=self.session, only_mask=True, post_process_mask=True)

        # Ensure only the largest connected component is returned
        mask = self.__find_largest_connected_component(mask)

        return mask

    def __find_largest_connected_component(self, mask):
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
        largest_component_mask = np.where(labels == largest_component_label, 255, 0).astype(
            np.uint8
        )

        return largest_component_mask
