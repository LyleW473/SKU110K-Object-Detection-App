import torch

from typing import Tuple

class Patchify:
    """
    Class for extracting patches from an image.
    """
    def __call__(self, image: torch.Tensor, patch_size:Tuple[int, int]) -> torch.Tensor:
        """
        Splits an image into patches of a specified size.
        - Returns the patches in the format (num_patches, channels, patch_height, patch_width)
        
        Args:
            image (torch.Tensor): The image to split into patches.
            patch_size (int): The size of the patches to create.

        Returns:
            torch.Tensor: A list of patches.
        """
        patch_height, patch_width = patch_size
        patches = []
        coords = []
        for j in range(0, image.shape[1], patch_height):
            for i in range(0, image.shape[2], patch_width):
                x1 = j
                x2 = x1 + patch_width
                y1 = i
                y2 = y1 + patch_height

                if x2 > image.shape[2]:
                    x2 = image.shape[2]
                    x1 = x2 - patch_width # Shift the start point to the left
                
                if y2 > image.shape[1]:
                    y2 = image.shape[1]
                    y1 = y2 - patch_height # Shift the start point upwards
            
                patch = image[:, y1:y2, x1:x2]
                start_coords = (x1, y1) # Save the coordinates of the patch
                patches.append(patch)
                coords.append(start_coords)
        return torch.stack(patches, dim=0), coords