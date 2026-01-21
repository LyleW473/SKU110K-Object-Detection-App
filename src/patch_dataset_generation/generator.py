import os
import pandas as pd
import numpy as np
import shutil
import time

from PIL import Image
from typing import Union, Tuple, List
from src.patch_dataset_generation.utils import visualise_image

class PatchDatasetGenerator:
    """
    Class for generating the patch dataset from the original dataset
    """
    def __init__(self, patch_size:int=224):
        """
        Initialises the PatchDatasetGenerator object.

        Args:
            patch_size (int): The size of the patches to generate, assumed to be square patches.
        """
        self.patch_size = patch_size # (patch_size x patch_size) patches
        self.stride = self.patch_size // 2 # Overlapping patches by half the patch size

    def _load_image(self, image_path:str) -> np.ndarray:
        """
        Load image from the given path and returns it as a
        numpy array in the format (H, W, C)

        Args:
            image_path (str): Path to the image file.
        """
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = np.array(img)
        return img
    
    def _find_bboxes_in_patch(self, patch:np.ndarray, annotations:pd.DataFrame, patch_x:int, patch_y:int) -> pd.DataFrame:
        """
        Finds the bounding boxes that are present in the given patch.

        Args:
            patch (np.ndarray): Patch as a numpy array in the format (H, W, C)
            annotations (pd.DataFrame): Annotations for the original image.
            patch_x (int): X-coordinate of the patch in the original image.
            patch_y (int): Y-coordinate of the patch in the original image.
        """
        H, W, _ = patch.shape
        annotations_copy = annotations.copy()
        patch_x_end = patch_x + W
        patch_y_end = patch_y + H

        # Vectorized check if any corner of the bounding box is inside the patch
        x_start_inside = (annotations_copy["x1"] >= patch_x) & (annotations_copy["x1"] < patch_x_end)
        x_end_inside = (annotations_copy["x2"] > patch_x) & (annotations_copy["x2"] <= patch_x_end)
        y_start_inside = (annotations_copy["y1"] >= patch_y) & (annotations_copy["y1"] < patch_y_end)
        y_end_inside = (annotations_copy["y2"] > patch_y) & (annotations_copy["y2"] <= patch_y_end)

        top_left_inside = (x_start_inside & y_start_inside)
        bottom_right_inside = (x_end_inside & y_end_inside)
        top_right_inside = (x_end_inside & y_start_inside)
        bottom_left_inside = (x_start_inside & y_end_inside)
        inside_patch = (top_left_inside | bottom_right_inside | top_right_inside | bottom_left_inside)

        # Filter annotations based on the vectorized check
        annotations_copy = annotations_copy[inside_patch]

        # Calculate the area of the original bounding boxes
        annotations_copy["bbox_area"] = (annotations_copy["x2"] - annotations_copy["x1"]) * (annotations_copy["y2"] - annotations_copy["y1"])

        # Adjust the bounding boxes to fit inside the patch
        annotations_copy["x1"] = annotations_copy["x1"].apply(lambda x: max(0, x - patch_x))
        annotations_copy["y1"] = annotations_copy["y1"].apply(lambda y: max(0, y - patch_y))
        annotations_copy["x2"] = annotations_copy["x2"].apply(lambda x: min(W, x - patch_x))
        annotations_copy["y2"] = annotations_copy["y2"].apply(lambda y: min(H, y - patch_y))

        # Calculate the proportion of the new bounding box area to the original bounding box area
        annotations_copy["new_bbox_area"] = (annotations_copy["x2"] - annotations_copy["x1"]) * (annotations_copy["y2"] - annotations_copy["y1"])
        annotations_copy["proportion"] = annotations_copy["new_bbox_area"] / annotations_copy["bbox_area"]

        # Filter out invalid bounding boxes
        valid_bboxes = (annotations_copy["bbox_area"] > 0) & (annotations_copy["new_bbox_area"] > 0) & (annotations_copy["proportion"] > 0.25)
        annotations_copy = annotations_copy[valid_bboxes]
        annotations_copy.drop(columns=["bbox_area", "new_bbox_area", "proportion"], inplace=True)

        # print(patch_x, patch_y, new_data.shape, annotations_copy.shape)
        return annotations_copy
    
    def _extract_patches(self, image:np.ndarray, annotations:pd.DataFrame, max_num_patches_per_image:int) -> Tuple[np.ndarray, List[pd.DataFrame]]:
        """
        Extracts patches from the given image, returning the patches as a single numpy array.
        - Finds all the possible patch co-ordinates first.
        - Randomly shuffles the patch coordinates.
        - Finds the bounding boxes present in each patch, exiting early if the number of patches processed
          exceeds the maximum number of patches per image.

        Args:
            image (np.ndarray): Image as a numpy array in the format (H, W, C)
            annotations (pd.DataFrame): Annotations for the image.
            max_num_patches_per_image (int): The maximum number of patch images to save/generate from
                                            each original image. If None, all patches are saved.
        """
        H, W, C = image.shape
        # visualise_image(image, annotations[["x1", "y1", "x2", "y2"]].values)

        # Find all possible patch coordinates
        patch_coords = []
        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):
                x_start = x
                y_start = y
                x_end = x + self.patch_size
                y_end = y + self.patch_size
                
                # Shift patch if it goes out of bounds
                if y_end > H:
                    diff = y_end - H
                    y_start -= diff
                    y_end -= diff
                
                if x_end > W:
                    diff = x_end - W
                    x_start -= diff
                    x_end -= diff
                patch_coords.append((x, y, x_end, y_end))

        # Shuffle patch coordinates to randomise the order
        indices = np.random.permutation(len(patch_coords))
        patch_coords = [patch_coords[i] for i in indices]

        # Find patches and annotations for each set of patch coordinates
        patches = []
        total_bboxes = 0
        annotations_for_each_patch = []
        for x_start, y_start, x_end, y_end in patch_coords:
            patch = image[y_start:y_end, x_start:x_end, :]
            bboxes_in_patch = self._find_bboxes_in_patch(patch=patch, annotations=annotations, patch_x=x_start, patch_y=y_start)

            num_bboxes_found = bboxes_in_patch.shape[0]
            if num_bboxes_found == 0: # Continue until we find a patch with bounding boxes or all patches are exhausted
                continue

            # visualise_image(patch, bboxes_in_patch[["x1", "y1", "x2", "y2"]].values)
            annotations_for_each_patch.append(bboxes_in_patch)
            patches.append(patch)
            total_bboxes += num_bboxes_found

            # Limit the number of patches per image
            if max_num_patches_per_image is not None:
                if len(patches) >= max_num_patches_per_image:
                    break

        # Check if no bounding boxes were found in any patch
        if total_bboxes == 0:
            return np.array([]), []
        
        # Stack all the patches to form a single numpy array
        patches = np.stack(patches, axis=0)
        return patches, annotations_for_each_patch
    
    def _choose_random_patches(
                            self, 
                            patches:np.ndarray, 
                            annotations_per_patch:List[pd.DataFrame],
                            num_patches:Union[int, None]) -> np.ndarray:
        """
        Randomly selects a given number of patches from the list of patches.
        - Used to limit the number of patches generated from each image.

        Args:
            patches (np.ndarray): List of patches as a numpy array.
            annotations_per_patch (List[pd.DataFrame]): List of dataframes containing the annotations for each patch.
            num_patches (Union[int, None]): Number of patches to select.
        """
        if num_patches is None:
            num_to_choose = len(patches)
        else:
            num_to_choose = min(num_patches, len(patches))
        indices = np.random.choice(len(patches), size=num_to_choose, replace=False)
        selected_patches = patches[indices]
        selected_annotations = [annotations_per_patch[i] for i in indices]
        return selected_patches, selected_annotations
    
    def generate(self, images_dir:str, annotations_path:str, max_num_patches_per_image:Union[int, None]=None):
        """
        Main method for generating the patch dataset.
        - Loads the images and annotations from the original dataset.
        - Converts the images into patches.
        - Saves the patches and their corresponding annotations.

        Args:
            images_dir (str): Path to the main directory containing the images.
            annotations_path (str): Path to the CSV file containing the annotations.
            max_num_patches_per_image (Union[int, None]): The maximum number of patch images to save/generate from
                                                        each original image. If None, all patches are saved.
        """

        all_images = os.listdir(images_dir)
        all_annotations = pd.read_csv(annotations_path)
        annotations_parent_dir = os.path.dirname(annotations_path)

        # Create directory for patches dataset
        parent_dir = os.path.dirname(images_dir)
        patches_dir = os.path.join(parent_dir, f"patches_dataset_{self.patch_size}")
        shutil.rmtree(patches_dir, ignore_errors=True) 
        os.makedirs(patches_dir, exist_ok=True)
        
        # Extract and save patches from original images
        all_new_annotations = {column: [] for column in all_annotations.columns}
        for image_num, image_file in enumerate(all_images):
            print(f"Processing Image {image_num+1}/{len(all_images)}: {image_file}, Progress: {((image_num+1)/len(all_images))*100:.4f}%")

            # Attempt to load image (Could be corrupted)
            try:
                img = self._load_image(image_path=f"{images_dir}/{image_file}")
            except:
                print(f"Error loading image: {image_file}")
                continue
            
            corresponding_annotations = all_annotations[all_annotations["image_name"] == image_file]
            patches, annotations_per_patch = self._extract_patches(
                                                                image=img, 
                                                                annotations=corresponding_annotations, 
                                                                max_num_patches_per_image=max_num_patches_per_image
                                                                )
            patches, annotations_per_patch = self._choose_random_patches(
                                                                        patches=patches, 
                                                                        annotations_per_patch=annotations_per_patch, 
                                                                        num_patches=max_num_patches_per_image
                                                                        )
            # Save patches
            original_image_name = image_file.split(".")[0] # E.g., "image_1.jpg" -> "image_1"
            for i, (patch, patch_annotations) in enumerate(zip(patches, annotations_per_patch)):
                patch_pil = Image.fromarray(patch)
                patch_save_name = f'{original_image_name}__patch_{i}.jpg'
                patch_save_path = os.path.join(patches_dir, patch_save_name)
                patch_pil.save(patch_save_path)

                # Over-write the image_name, image_width, and image_height columns (for the patch)
                patch_annotations["image_name"] = patch_save_name
                patch_annotations["image_width"] = self.patch_size
                patch_annotations["image_height"] = self.patch_size
                for column in all_new_annotations:
                    all_new_annotations[column].extend(patch_annotations[column].values)

        # Save the new annotations after each image
        annotations_df = pd.DataFrame(all_new_annotations)
        annotations_df.to_csv(os.path.join(annotations_parent_dir, f"annotations_patches_{self.patch_size}.csv"), index=False)