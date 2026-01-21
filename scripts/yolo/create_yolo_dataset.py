import set_path
import sys
from src.yolo_data_prep.generator import YOLODataGenerator

if __name__ == "__main__":

    annotations_dir = "data/annotations"

    chosen_option = "patches_640"
    options = {
        # (Base dataset with no custom data splitting)
        "base": {
            "train": "annotations_train.csv", 
            "val": "annotations_val.csv", 
            "test": "annotations_test.csv", 
            "images_dir": "data/filtered_images"
            },
        # Custom data splitting
        "custom": {
            "train": "annotations_yolo_train.csv", 
            "val": "annotations_yolo_val.csv", 
            "test": "annotations_yolo_test.csv",
            "images_dir": "data/yolo_split_images_normal" # Use the normal images for the patches
            },
        "patches_640": {
            "train": "annotations_yolo_train.csv", 
            "val": "annotations_yolo_val.csv", 
            "test": "annotations_yolo_test.csv",
            "images_dir": "data/yolo_split_images_patches_640" # Use the normal images for the patches
            },
        }
    new_base_dataset_path = f"data/yolo_dataset_{chosen_option}"
    generator = YOLODataGenerator(new_base_dataset_path=new_base_dataset_path)

    if generator.check_dataset_exists():
        print("Dataset already exists, Terminating...")
        sys.exit(0)

    generator.create_yolo_directories()

    train_csv = generator.load_csv(f"{annotations_dir}/{options[chosen_option]['train']}")
    val_csv = generator.load_csv(f"{annotations_dir}/{options[chosen_option]['val']}")
    test_csv = generator.load_csv(f"{annotations_dir}/{options[chosen_option]['test']}")
    images_dir = options[chosen_option]["images_dir"]

    generator.create_dataset(train_csv=train_csv, val_csv=val_csv, test_csv=test_csv, images_dir=images_dir)
    generator.create_yaml()