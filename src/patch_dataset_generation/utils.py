import cv2
import numpy as np

from PIL import Image
from typing import List, Tuple

def visualise_image(image:np.ndarray, bboxes:List[Tuple[int, int, int, int]]) -> None:
    """
    Visualises the image with bounding boxes drawn around the objects.

    Args:
        image (np.ndarray): The image array to visualise.
        bboxes (List[Tuple[int, int, int, int]]): The bounding boxes for the objects
                                                  to draw on the image.
    """
    image_copy = image.copy()
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        image_copy = cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    image_pil = Image.fromarray(image_copy)
    image_pil.show()