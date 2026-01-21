import set_path
import os
import numpy as np
import random

from src.patch_dataset_generation.generator import PatchDatasetGenerator

if __name__ == "__main__":

    # Reproducibility
    seed = 2004
    np.random.seed(seed)
    random.seed(seed)

    PATCH_SIZE = 640
    PDG = PatchDatasetGenerator(patch_size=PATCH_SIZE)

    images_dir = "data/renamed_dataset"
    annotations_path = "data/annotations/annotations_all.csv"

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}, please run 'rename_data.py' first.")

    max_num_patches_per_image = 1
    PDG.generate(
                images_dir=images_dir, 
                annotations_path=annotations_path,
                max_num_patches_per_image=max_num_patches_per_image,
                )