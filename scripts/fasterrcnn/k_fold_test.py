import set_path
import copy
import torch
import numpy as np
import random

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou, nms

from src.training.fasterrcnn.dataset import CustomDataset, custom_collate_fn, PatchesDataset
from src.training.fasterrcnn.engine import calculate_tp_fn_fp
from src.training.fasterrcnn.data import (
                                        create_processed_dataframes, 
                                        load_annotations, 
                                        train_test_split,
                                        load_dataloader
                                        )
from src.training.fasterrcnn.transforms import construct_image_transforms
from src.training.fasterrcnn.utils import load_environment_variables, initialise_clearml_task, get_image_paths, calculate_dataset_statistics, save_image
from src.training.fasterrcnn.patchify import Patchify

if __name__ == "__main__":

    # Set deterministic behaviour for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    random.seed(42)
    
    """
    Should only be used for the Faster R-CNN model
    when trained on patch images of size (224x224)
    or (512x512). When training on the full images,
    the testing is already done in the 'train_model.py'.
    It can however be used to 
    """
    # Constants
    chosen_option = "patch_512"
    options = {
            "patch_224": {
                "patch_size": 224
                },
            "patch_512": {
                "patch_size": 512
                },
            "base": {
                "patch_size": None # Will use (512x512) patches at inference, but the way data is split is different
                }
            }

    # Constants
    PATCH_SIZE = options[chosen_option]["patch_size"]
    NUM_CLASSES = 2

    # Always test on the full images:
    ANNOTATIONS_PATH = "data/annotations/annotations_all.csv"
    IMAGE_DIR = "data/renamed_dataset/"
    
    PRINT_INTERVAL = 1 # (Batches)
    MODEL_DIR = "saved_models"
    MODEL_NUM = 71
    
    # Define configuration parameters
    config_params = {
                    'learning-rate': 1e-3,
                    'batch-size': 4, 
                    'num-epochs': 10,
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
    if PATCH_SIZE is None: # Set to use 512x512 patches for testing models that were trained on RandomCrops
        PATCH_SIZE = 512
    # Remove any combined features that only appear once (As they cannot be split using StratifiedShuffleSplit)
    value_counts = image_statistics["combined_feature"].value_counts()
    image_statistics = image_statistics[image_statistics["combined_feature"].isin(value_counts[value_counts > 1].index)]
    image_statistics = image_statistics.reset_index(drop=True)
    _, test_df, train_image_names, test_image_names = train_test_split(annotations_df, image_statistics, test_size=0.2)
    
    # Use custom dataset statistics for training (only using training images)
    image_paths = get_image_paths(IMAGE_DIR)
    set_train_image_names = set(train_image_names)
    print(len(image_paths))
    image_paths = [path for path in image_paths if path.split("/")[-1] in set_train_image_names]
    print(len(image_paths), len(train_image_names))
    mean, std = calculate_dataset_statistics(image_paths, chosen_option="patch_512")
    print(mean, std)
    
    # Get the image transforms (data augmentation)
    image_transforms = construct_image_transforms(chosen_option="patch_512")

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
    task = initialise_clearml_task(
                                    configuration_params=config_params,
                                    task_name="K-Fold Cross Validation Inference (Faster R-CNN)"
                                    )
    
    test_dataset = CustomDataset(test_df, IMAGE_DIR, transforms=image_transforms)
    clone_config_params = copy.deepcopy(config_params)
    clone_config_params["batch-size"] = 1 # Using full images, so batch size is 1
    test_dl = load_dataloader(
                            dataset=test_dataset, 
                            config_params=clone_config_params,
                            collate_fn=custom_collate_fn,
                            shuffle=False # Do not shuffle the test set
                            )

    # Test model
    k_fold_results = {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0, "mean_iou": 0.0, "mean_precision": 0.0, "mean_recall": 0.0, "f1_score": 0.0}
    patchify = Patchify()
    for k in range(1, 6):
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, **custom_model_args)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
        model.to(device)
        model.eval()

        model_state_dict = torch.load(f"{MODEL_DIR}/model_{MODEL_NUM}/fold_{k}.pt", map_location=device)
        model.load_state_dict(model_state_dict)

        image_results = {k: 0.0 for k in k_fold_results.keys()}
        for i, (images, target) in enumerate(test_dl):
            images = [img.to(device) for img in images]
            target = [{k: v.cpu() for k, v in t.items()} for t in target]

            full_image = images[0] # Only one image in the batch, unpack it
            full_target = target[0] # Only one target in the batch, unpack it

            # Get the patches
            patches, coords = patchify(full_image, patch_size=(PATCH_SIZE, PATCH_SIZE))

            # Aggregate predictions across all patches
            mini_dataloader = load_dataloader(
                                            dataset=PatchesDataset(patches, coords),
                                            config_params=config_params,
                                            collate_fn=custom_collate_fn,
                                            shuffle=False
                                            )
            all_preds = {
                        "boxes": torch.zeros(0, 4, dtype=torch.float32), 
                        "labels": torch.zeros(0, dtype=torch.int64), 
                        "scores": torch.zeros(0, dtype=torch.float32)
                        }
            for j, (patch_images, coords) in enumerate(mini_dataloader):
                patch_images = [img.to(device) for img in patch_images]
                coords = [coord for coord in coords]

                with torch.no_grad():
                    outputs = model(patch_images)
                    outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

                    for x in range(len(outputs)):
                        # Add starting co-ordinates to the boxes
                        outputs[x]["boxes"][:, 0] += coords[x][0]
                        outputs[x]["boxes"][:, 1] += coords[x][1]
                        outputs[x]["boxes"][:, 2] += coords[x][0]
                        outputs[x]["boxes"][:, 3] += coords[x][1]

                        # Append the all_preds
                        all_preds["boxes"] = torch.cat((all_preds["boxes"], outputs[x]["boxes"]), dim=0)
                        all_preds["labels"] = torch.cat((all_preds["labels"], outputs[x]["labels"]), dim=0)
                        all_preds["scores"] = torch.cat((all_preds["scores"], outputs[x]["scores"]), dim=0)
                        # print(all_preds["boxes"].shape, all_preds["labels"].shape, all_preds["scores"].shape)

            # save_image(full_image, full_target, "target_image.png")
            # save_image(full_image, all_preds, "pred_image.png")

            # Perform NMS to remove overlapping boxes
            indices = nms(all_preds["boxes"], all_preds["scores"], iou_threshold=0.1)
            all_preds["boxes"] = all_preds["boxes"][indices]
            all_preds["labels"] = all_preds["labels"][indices]
            all_preds["scores"] = all_preds["scores"][indices]
            # save_image(full_image, all_preds, "nms_image.png")

            # Calculate metrics
            map_metric = MeanAveragePrecision()
            map_metric.warn_on_many_detections = False # Suppress warning
            map_metric.update([all_preds], target)
            # print(all_preds["boxes"].dtype, all_preds["labels"].dtype, all_preds["scores"].dtype)
            mAP_all_preds = map_metric.compute()
            # print(f"Fold {k} Image {i} mAP all_preds: {mAP_all_preds}")
            # print()


            pred_boxes, pred_labels = all_preds["boxes"], all_preds["labels"]
            target_boxes, target_labels = full_target["boxes"], full_target["labels"]
            iou = box_iou(pred_boxes, target_boxes)

            if len(pred_boxes) == 0 or len(target_boxes) == 0:
                tp = 0
                fp = 0
                fn = len(target_boxes) # Meaning that we missed all the GT boxes
            else:
                tp, fp, fn = calculate_tp_fn_fp(
                                                iou=iou,
                                                pred_boxes=pred_boxes,
                                                pred_labels=pred_labels,
                                                target_boxes=target_boxes,
                                                target_labels=target_labels,
                                                iou_threshold=0.5
                                                )
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            mean_iou = iou.mean().item()
            # print(mAP_all_preds.keys())

            image_results["mAP"] += mAP_all_preds["map"]
            image_results["mAP_50"] += mAP_all_preds["map_50"]
            image_results["mAP_75"] += mAP_all_preds["map_75"]
            image_results["mean_iou"] += mean_iou
            image_results["mean_precision"] += precision
            image_results["mean_recall"] += recall
            image_results["f1_score"] += f1_score

            # Printing metrics for this image
            message = (
                    f"Fold: {k} | "
                    f"Image: {i+1}/{len(test_dl)} | "
                    f"mAP: {mAP_all_preds['map']:.4f} | "
                    f"mAP_50: {mAP_all_preds['map_50']:.4f} | "
                    f"mAP_75: {mAP_all_preds['map_75']:.4f} | "
                    f"Mean IoU: {mean_iou:.4f} | "
                    f"Mean Precision: {precision:.4f} | "
                    f"Mean Recall: {recall:.4f} | "
                    f"F1 Score: {f1_score:.4f} | "
                    f"Progress: {((i + 1) / len(test_dl) * 100):.5f}%"
                    )
            print(message)
        
        # Average the results for all images
        image_results = {k: v / len(test_dl) for k, v in image_results.items()}
        for key, value in image_results.items():
            k_fold_results[key] += value
        print(f"Fold {k} Results: {image_results}")
    
    k_fold_results = {k: v / 5 for k, v in k_fold_results.items()}
    print(f"K-Fold Results: {k_fold_results}")