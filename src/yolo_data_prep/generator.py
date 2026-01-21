import os
import pandas as pd
import shutil
import yaml

class YOLODataGenerator:
    """
    Class for creating a dataset for YOLO models.
    """
    def __init__(self, new_base_dataset_path:str):
        """
        Args:
            new_base_dataset_path (str): The path to the new base dataset.
        """
        self.new_base_dataset_path = new_base_dataset_path

    def check_dataset_exists(self) -> bool:
        """
        Checks if the dataset directories for the YOLO models exist.
        """
        train_exists = os.path.exists(f"{self.new_base_dataset_path}/train")
        val_exists = os.path.exists(f"{self.new_base_dataset_path}/val")
        test_exists = os.path.exists(f"{self.new_base_dataset_path}/test")

        if not (train_exists and val_exists and test_exists):
            return False
        
        n_train_images = len(os.listdir(f"{self.new_base_dataset_path}/train/images"))
        n_val_images = len(os.listdir(f"{self.new_base_dataset_path}/val/images"))
        n_test_images = len(os.listdir(f"{self.new_base_dataset_path}/test/images"))

        n_train_labels = len(os.listdir(f"{self.new_base_dataset_path}/train/labels"))
        n_val_labels = len(os.listdir(f"{self.new_base_dataset_path}/val/labels"))
        n_test_labels = len(os.listdir(f"{self.new_base_dataset_path}/test/labels"))

        if not (n_train_images == n_train_labels):
            return False
        if not (n_val_images == n_val_labels):
            return False
        if not (n_test_images == n_test_labels):
            return False
        return True
        
    def create_yolo_directories(self) -> None:
        """
        Creates the directories for the dataset for YOLO models.
        """
        for directory in ["train", "val", "test"]:
            os.makedirs(f"{self.new_base_dataset_path}/{directory}")
            os.makedirs(f"{self.new_base_dataset_path}/{directory}/images")
            os.makedirs(f"{self.new_base_dataset_path}/{directory}/labels")

    def load_csv(self, csv_path:str):
        """
        Load the CSV file containing the annotations.
        
        Args:
            csv_path (str): The path to the CSV file.
        """
        csv = pd.read_csv(csv_path)
        return csv
    
    def prepare_csv(self, csv:pd.DataFrame):
        """
        Prepares the CSV file for creating the YOLO dataset.
        - Adds the normalised bounding box annotations required for YOLO models.

        Args:
            csv (pd.DataFrame): The CSV file containing the annotations.
        """
        column_names = ["image_name", "x1", "y1", "x2", "y2", "class", "image_width", "image_height"]
        csv.columns = column_names

        # Add bbox columns
        csv["center_x"] = (csv["x1"] + csv["x2"]) / 2
        csv["center_y"] = (csv["y1"] + csv["y2"]) / 2
        csv["width"] = csv["x2"] - csv["x1"]
        csv["height"] = csv["y2"] - csv["y1"]
        
        # Normalise
        csv["center_x"] = csv["center_x"] / csv["image_width"]
        csv["center_y"] = csv["center_y"] / csv["image_height"]
        csv["width"] = csv["width"] / csv["image_width"]
        csv["height"] = csv["height"] / csv["image_height"]

        # Remove columns that are not needed
        csv = csv[["image_name", "center_x", "center_y", "width", "height", "class"]]
        return csv

    def create_dataset(self, train_csv:pd.DataFrame, val_csv:pd.DataFrame, test_csv:pd.DataFrame, images_dir:str):
        """
        Creates a YOLO dataset from the training, validation, and test CSV files and the images directory.

        Args:
            train_csv (pd.DataFrame): The CSV file containing the training annotations.
            val_csv (pd.DataFrame): The CSV file containing the validation annotations.
            test_csv (pd.DataFrame): The CSV file containing the test annotations.
            images_dir (str): The directory containing all of the images for all splits.
        """
        class_mappings = {"object": 1}
        train_csv = self.prepare_csv(train_csv)
        val_csv = self.prepare_csv(val_csv)
        test_csv = self.prepare_csv(test_csv)

        all_image_names = os.listdir(images_dir)
        for i, image_name in enumerate(all_image_names):
            print(f"Processing image {i + 1}/{len(all_image_names)} | Progress: {((i + 1) / len(all_image_names)) * 100:.5f}%")
            image_id = image_name.split(".")[0]
            image_path = f"{images_dir}/{image_name}"

            if image_name.startswith("train"):
                selected_csv = train_csv
                selected_dir = "train"
            elif image_name.startswith("val"):
                selected_csv = val_csv
                selected_dir = "val"
            elif image_name.startswith("test"):
                selected_csv = test_csv
                selected_dir = "test"
            else:
                raise ValueError("Unexpected image name")

            # Get the annotations for the image
            image_annotations = selected_csv[selected_csv["image_name"] == image_name]

            # Create annotations
            with open(f"{self.new_base_dataset_path}/{selected_dir}/labels/{image_id}.txt", "w") as f:
                if len(image_annotations) == 0:
                    f.write("")
                    continue

                for index, row in image_annotations.iterrows():
                    class_name = row["class"]
                    class_id = class_mappings[class_name]
                    center_x = row["center_x"]
                    center_y = row["center_y"]
                    width = row["width"]
                    height = row["height"]

                    f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")

            # Copy image to the correct directory
            shutil.copy(image_path, f"{self.new_base_dataset_path}/{selected_dir}/images")

    def create_yaml(self):
        """
        Creates the yaml file for the dataset configuration.
        """
        abs_path = os.path.abspath(self.new_base_dataset_path)
        dataset_config = {
                        "path": abs_path,
                        "train": "train",
                        "val": "val",
                        "test": "test",
                        "names": {
                                0: "object"
                                }
                        }
        with open(f"{self.new_base_dataset_path}/dataset.yaml", "w") as f:
            yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)