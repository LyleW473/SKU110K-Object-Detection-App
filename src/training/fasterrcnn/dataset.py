import torch
import os
import numpy as np
import pandas as pd
import albumentations

from PIL import Image
from torch.utils.data import Dataset
from typing import List, Tuple, Union, Optional, Dict

def custom_collate_fn(batch:List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Custom collate function for the FasterRCNN model.
    - Used to collate the image and target dictionaries into a single batch.
    """
    return tuple(zip(*batch))

class PatchesDataset(Dataset):
    """
    The dataset class for returning the patches and their
    starting coordinates.

    - Used at testing time for the FasterRCNN model.
    """
    def __init__(self, patches:torch.Tensor, coords:List[Tuple[float, float]]):
        """
        Initialises a PatchesDataset object.

        Args:
            patches (torch.Tensor): The patches tensor.
            coords (List[Tuple[float, float]]): The starting coordinates of the patches, in 
                                                the same order as the patches tensor. This will
                                                be used to reconstruct the original image.
        """
        self.patches = patches
        self.coords = coords
    
    def __len__(self) -> int:
        """
        Defines the length of the dataset.
        """
        return len(self.patches)
    
    def __getitem__(self, i:int) -> Tuple[torch.Tensor, Tuple[float, float]]:
        """
        Returns the patch and the starting coordinates of the patch at index i.

        Args:
            i (int): The index of the patch and starting coordinates to return.
        """
        return self.patches[i], self.coords[i]
    
class CustomDataset(Dataset):
    """
    Custom dataset class for the FasterRCNN model that is used to
    train the model on the base images or the patch images.
    """
    def __init__(
            self, 
            annotations_df:pd.DataFrame, 
            image_dir:str, 
            transforms:Optional[Union[albumentations.Compose, None]]=None):
        """
        Intiailises a CustomDataset object.

        Args:
            annotations_df (pd.DataFrame): The dataframe containing the annotations for the images.
            image_dir (str): The directory containing the images.
            transforms (Union[albumentations.Compose, None], optional): The transformations to apply to the images. Defaults to None.
        """
        super().__init__()
        self.image_data = annotations_df.groupby('image_name')
        self.image_dir = image_dir
        self.transforms = transforms

    def __len__(self) -> int:
        """
        Defines the length of the dataset.
        """
        return len(self.image_data)
    
    def __getitem__(self, i:int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns the image and the target dictionary for the image at index i.

        Args:
            i (int): The index of the image to return.
        """
        image_name = list(self.image_data.groups.keys())[i]
        records = self.image_data.get_group(image_name)

        image_path = os.path.join(self.image_dir, image_name)
        image = np.array(Image.open(image_path).convert("RGB"))
        boxes = records[['x1', 'y1', 'x2', 'y2']].values
        labels = [1 for _ in range(len(boxes))]
        
        if self.transforms:
            H, W, C = image.shape

            # Clipping the boxes to the image dimensions
            boxes[:, 0] = np.clip(boxes[:, 0], a_min=0, a_max=W)
            boxes[:, 1] = np.clip(boxes[:, 1], a_min=0, a_max=H)
            boxes[:, 2] = np.clip(boxes[:, 2], a_min=0, a_max=W)
            boxes[:, 3] = np.clip(boxes[:, 3], a_min=0, a_max=H)

            # Data augmentation (No normalisation, that is done internally by FasterRCNN)
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]
        
        # [0, 255] -> [0, 1]
        image = image / 255.0

        # (H, W, C) -> (C, H, W)
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)

        target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                }
        return image, target