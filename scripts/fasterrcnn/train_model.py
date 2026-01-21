import set_path
import copy
import os
import torch
import numpy as np
import random

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights

from src.training.fasterrcnn.dataset import CustomDataset, custom_collate_fn
from src.training.fasterrcnn.engine import train_model_k_folds, evaluate_on_test_set
from src.training.fasterrcnn.data import (
                                        create_processed_dataframes, 
                                        load_annotations, 
                                        train_test_split,
                                        create_train_val_folds,
                                        create_train_val_dataloaders,
                                        load_dataloader
                                        )
from src.training.fasterrcnn.transforms import construct_image_transforms
from src.training.fasterrcnn.utils import load_environment_variables, initialise_clearml_task, get_image_paths, calculate_dataset_statistics

if __name__ == "__main__":

    # Set deterministic behaviour for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    random.seed(42)

    # Constants
    chosen_option = "base"
    options = {
            "patch_224": {
                "annotations_path": "data/annotations/annotations_patches_224.csv", 
                "image_dir": "data/patches_dataset_224/",
                "patch_size": 224
                },
            "patch_512": {
                "annotations_path": "data/annotations/annotations_patches_512.csv", 
                "image_dir": "data/patches_dataset_512/",
                "patch_size": 512
                },
            "base": {
                "annotations_path": "data/annotations/annotations_all.csv", 
                "image_dir": "data/renamed_dataset/",
                "patch_size": None # Will use (512x512) patches at inference, but the way data is split is different
                }
            }

    # Constants
    PATCH_SIZE = options[chosen_option]["patch_size"]
    NUM_CLASSES = 2
    ANNOTATIONS_PATH = options[chosen_option]["annotations_path"]
    IMAGE_DIR = options[chosen_option]["image_dir"]
    PRINT_INTERVAL = 1 # (Batches)
    
    # Define configuration parameters
    config_params = {
                    'learning-rate': 1e-3,
                    'batch-size': 4, 
                    'num-epochs': 5,
                    'momentum': 0.9,
                    }

    annotations_df = load_annotations(ANNOTATIONS_PATH)
    image_names_values = annotations_df["image_name"].value_counts().values
    max_num_objects = image_names_values.max()

    if PATCH_SIZE == 224 or PATCH_SIZE is None:
        mean_num_objects = image_names_values.mean()
        std_num_objects = image_names_values.std()
        num_objects_bins = [
                            0,
                            max(0, mean_num_objects - 2 * std_num_objects),
                            max(0, mean_num_objects - std_num_objects),
                            max(0, mean_num_objects),
                            max(0, mean_num_objects + std_num_objects),
                            max(0, mean_num_objects + 2 * std_num_objects),
                            max(0, max_num_objects)
                            ]
        num_objects_bins = list(set(num_objects_bins))

    elif PATCH_SIZE == 512:
        num_bins = 4 # 4 bins for ["Few", "Moderate", "Above Average", "Many"], ["Small", "Medium", "Large", "Very Large"]
        counts, bin_edges = np.histogram(image_names_values, bins=num_bins)
        num_objects_bins = list(bin_edges)
    else:
        raise ValueError("Unsupported patch size. Please use 224 or 512.")
    
    num_objects_bins.sort()
    print(num_objects_bins)
    annotations_df, image_statistics = create_processed_dataframes(
                                                                annotations_df=annotations_df,
                                                                num_objects_bins=num_objects_bins,
                                                                use_categories=(PATCH_SIZE == 512)
                                                                )

    train_df, test_df, train_image_names, test_image_names = train_test_split(annotations_df, image_statistics, test_size=0.2)

    all_splits = create_train_val_folds(
                                        annotations_df=annotations_df, 
                                        image_statistics=image_statistics, 
                                        train_image_names=train_image_names
                                        )
    
    # Use custom dataset statistics for training (only using training images)
    image_paths = get_image_paths(IMAGE_DIR)
    set_train_image_names = set(train_image_names)
    print(len(image_paths))
    image_paths = [path for path in image_paths if path.split("/")[-1] in set_train_image_names]
    print(len(image_paths), len(train_image_names))
    mean, std = calculate_dataset_statistics(image_paths, chosen_option=chosen_option)
    print(mean, std)
    
    # Sanity checks:
    for i in range(5):
        print(i, all_splits[i]["train"].shape, all_splits[i]["val"].shape)
        assert (all_splits[i]["train"].shape[0] + all_splits[i]["val"].shape[0] + test_df.shape[0]) == annotations_df.shape[0], "Data processing incorrect, missing objects."
    
    # Get the image transforms (data augmentation)
    image_transforms = construct_image_transforms(chosen_option=chosen_option)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Define model 
    custom_model_args = {
                        # "min_size": PATCH_SIZE,
                        # "max_size": PATCH_SIZE,
                        "image_mean": mean,
                        "image_std": std,
                        "box_detections_per_img": max_num_objects, # Default is 100
                        "box_score_thresh": 0.75, # Default is 0.05
                        }
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, **custom_model_args)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    # Move the model to the device
    model.to(device)

    load_environment_variables(env_path='./.env')
    task = initialise_clearml_task(configuration_params=config_params)

    # # Log the model
    # TODO: Unsure if this is necessary?
    # dataset = ClearMLDataset.get(dataset_project="Object Detection Project", dataset_name="Patches Dataset")
    # dataset_path = dataset.get_local_copy()

    # Create a directory to save the model
    os.makedirs("saved_models", exist_ok=True)
    num_models = len(os.listdir("saved_models"))
    model_dir = f"saved_models/model_{num_models}"
    os.makedirs(model_dir)

    # Load the dataloaders
    dataloaders_train, dataloaders_val = create_train_val_dataloaders(
                                                                    all_splits=all_splits, 
                                                                    image_dir=IMAGE_DIR, 
                                                                    image_transforms=image_transforms, 
                                                                    config_params=config_params,
                                                                    collate_fn=custom_collate_fn,
                                                                    )
    
    test_dataset = CustomDataset(test_df, IMAGE_DIR, transforms=image_transforms)
    test_dl = load_dataloader(
                            dataset=test_dataset, 
                            config_params=config_params,
                            collate_fn=custom_collate_fn,
                            shuffle=False # Do not shuffle the test set
                            )
    
    # TEMP: Print the first 5 batches of the test loader#
    print("num batches in dataloader", len(test_dl))
    for i, (images, targets) in enumerate(test_dl):
        print(f"Batch {i}: {len(images)} images, {len(targets)} targets")
        print(targets)
        print(i)
        if i > 5:
            break
        
    # Start training
    best_model, best_fold_model_path = train_model_k_folds(
                                                        model=model, 
                                                        train_dataloaders=dataloaders_train,
                                                        val_dataloaders=dataloaders_val,
                                                        device=device, 
                                                        config_params=config_params,
                                                        model_dir=model_dir,
                                                        print_interval=PRINT_INTERVAL
                                                        )


    # Test model
    final_model = copy.deepcopy(best_model)
    final_model.load_state_dict(torch.load(best_fold_model_path, weights_only=True)) # Adjust this path if you want to load the overall best fold

    # Call the test function to evaluate on the test set
    test_results = evaluate_on_test_set(final_model, test_dl, device)

    # Log the test results (e.g., mAP, Precision, Recall) to ClearML
    task.upload_artifact(name="Test Results", artifact_object=test_results)

    task.close() # uncomment to close the task