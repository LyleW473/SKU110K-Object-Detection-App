import set_path
import sys
from src.yolo_data_prep.splitter import YOLODataSplitter

if __name__ == "__main__":
    splitter = YOLODataSplitter()

    MAIN_DATASET_DIR = "data"
    options = {
                "normal": {
                    "annotations_path": f"{MAIN_DATASET_DIR}/annotations/annotations_all.csv", 
                    "images_dir": f"{MAIN_DATASET_DIR}/renamed_dataset"
                    },
                "patches_640": {
                            "annotations_path": f"{MAIN_DATASET_DIR}/annotations/annotations_patches_640.csv", 
                            "images_dir": f"{MAIN_DATASET_DIR}/patches_dataset_640"
                            },
                }
    # Change this to "base" if you want to use the base dataset, the images are resized to 640x640 when training though.
    chosen_option = "patches_640"


    annotations_path = options[chosen_option]["annotations_path"]
    images_dir = options[chosen_option]["images_dir"]
    new_images_dir = f"{MAIN_DATASET_DIR}/yolo_split_images_{chosen_option}"
    splitter.split_data(annotations_path=annotations_path, images_dir=images_dir, new_images_dir=new_images_dir)