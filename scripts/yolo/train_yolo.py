import set_path
import os
import yaml
import torch
import matplotlib
import torch.version
matplotlib.use('Agg') # Use non-interactive backend to prevent plots from showing up
from clearml import Task
from ultralytics import YOLO

if __name__ == "__main__":
    task = Task.init(project_name="Object Detection Project", task_name="train_yolo")

    chosen_option = "patches_640"
    options = set(["base", "custom", "patches_640"])
    dataset_dir = f"data/yolo_dataset_{chosen_option}"

    model_variant = "yolo11n"
    task.set_parameter("model_variant", model_variant)

    model = YOLO(f"{model_variant}.pt")
    print(torch.cuda.is_available(), torch.cuda.current_device(), torch.version.cuda)

    hyperparameters = {
                    "epochs": 10,
                    "batch": 16,
                    "imgsz": 640,
                    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    "single_cls": True, # There is only one class in this dataset
                    }
    task.connect(model) # Connect to ClearML task

    dataset_yaml_path = f"{dataset_dir}/dataset.yaml"
    
    if not os.path.exists("runs/detect/tune/best_hyperparameters.yaml"):
        # Conduct hyperparameter tuning to find the best hyperparameters
        model.tune(
                    data=dataset_yaml_path,
                    iterations=5,
                    optimizer="AdamW",

                    # Skip plots, checkpoints and validation until last epoch
                    plots=True,
                    save=True,
                    val=True,

                    # Base training hyperparameters
                    **hyperparameters
                    )
    else:
        # Load best hyperparameters from tuning
        with open("runs/detect/tune/best_hyperparameters.yaml", "r") as file:
            best_hyperparameters = yaml.load(file, Loader=yaml.FullLoader)
        hyperparameters.update(best_hyperparameters)
        print("Updated hyperparameters:", hyperparameters)
        results = model.train(data=dataset_yaml_path, **hyperparameters)