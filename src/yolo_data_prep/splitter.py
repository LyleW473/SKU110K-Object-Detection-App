import os
import shutil
import pandas as pd

from src.training.fasterrcnn.data import load_annotations, create_processed_dataframes, train_test_split
from typing import List

class YOLODataSplitter:
    """
    Class for splitting the original dataset into train, val, and test sets for YOLO models.
    """
    def split_data(self, annotations_path:str, images_dir:str, new_images_dir:str):
        """
        Splits the original dataset into train, val, and test sets for YOLO models.

        Args:
            annotations_path (str): The path to the annotations CSV file.
            images_dir (str): The directory containing the images.
        """
        annotations = load_annotations(annotations_path)
        
        image_names_values = annotations["image_name"].value_counts().values
        mean_num_objects = image_names_values.mean()
        std_num_objects = image_names_values.std()
        max_num_objects = image_names_values.max()
        num_objects_bins = [
                            0,
                            mean_num_objects - 2 * std_num_objects,
                            mean_num_objects - std_num_objects,
                            mean_num_objects,
                            mean_num_objects + std_num_objects,
                            mean_num_objects + 2 * std_num_objects,
                            max_num_objects
                            ]
        num_objects_bins = list(set(num_objects_bins))
        num_objects_bins.sort()
        annotations_df, image_statistics = create_processed_dataframes(annotations_df=annotations, num_objects_bins=num_objects_bins)

        # Train + Test set
        training_df, test_df, train_image_names, test_image_names = train_test_split(annotations_df, image_statistics, test_size=0.2)

        # Train + Val set
        train_image_statistics = image_statistics[image_statistics["image_name"].isin(train_image_names)]

        # Reset indices for splitting again
        training_df = training_df.sort_values(by="image_name")
        train_image_statistics = train_image_statistics.sort_values(by="image_name")
        train_image_statistics.reset_index(drop=True, inplace=True)
        training_df.reset_index(drop=True, inplace=True)

        train_df, val_df, train_image_names, val_image_names = train_test_split(training_df, train_image_statistics, test_size=0.2)
        assert len(train_df) + len(val_df) + len(test_df) == len(annotations_df), "Annotations are not split correctly"
        assert len(train_df["image_name"].value_counts()) == len(train_image_names), "Train annotations are not split correctly"
        assert len(val_df["image_name"].value_counts()) == len(val_image_names), "Val annotations are not split correctly"
        assert len(test_df["image_name"].value_counts()) == len(test_image_names), "Test annotations are not split correctly"

        # Rename all images (copies to new directory)
        os.makedirs(new_images_dir)
        self.rename_images("train", train_image_names, train_df, images_dir, new_images_dir)
        self.rename_images("val", val_image_names, val_df, images_dir, new_images_dir)
        self.rename_images("test", test_image_names, test_df, images_dir, new_images_dir)

        # Save annotations
        train_df = train_df.drop(columns=["bbox_area"])
        val_df = val_df.drop(columns=["bbox_area"])
        test_df = test_df.drop(columns=["bbox_area"])
        train_df.to_csv(f"{os.path.dirname(annotations_path)}/annotations_yolo_train.csv", index=False)
        val_df.to_csv(f"{os.path.dirname(annotations_path)}/annotations_yolo_val.csv", index=False)
        test_df.to_csv(f"{os.path.dirname(annotations_path)}/annotations_yolo_test.csv", index=False)

    def rename_images(self, mode:str, image_names:List[str], annotations_df:pd.DataFrame, images_dir:str, new_images_dir:str):
        """
        Renames the images to a new directory with the specified mode.
        - E.g., "image_2323.jpg" -> "train_0.jpg"
        - All images are stored in the same directory, which is the 'new_images_dir'.

        Args:
            mode (str): The mode of the images (train, val, test).
            image_names (List[str]): The list of image names to rename.
            annotations_df (pd.DataFrame): The annotations dataframe containing the image names.
            images_dir (str): The directory containing the original images.
            new_images_dir (str): The directory to store the renamed images.
        """
        for i, image_name in enumerate(image_names):
            print(f"Mode: {mode} | Processing image {i + 1}/{len(image_names)} | Progress: {((i + 1) / len(image_names)) * 100:.5f}%")
            image_path = f"{images_dir}/{image_name}" # E.g., "data/renamed_images/image_2323.jpg"
            new_image_path = f"{new_images_dir}/{mode}_{i}.jpg" # E.g., "data/yolo_split_images/train_0.jpg"
            annotations_df.loc[annotations_df["image_name"] == image_name, "image_name"] = f"{mode}_{i}.jpg" # Replace image name in annotations

            assert os.path.exists(image_path), f"Image path {image_path} does not exist"
            shutil.copy(image_path, new_image_path)