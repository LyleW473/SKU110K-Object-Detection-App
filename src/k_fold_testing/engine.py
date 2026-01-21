import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou, nms

from src.training.fasterrcnn.dataset import CustomDataset, custom_collate_fn, PatchesDataset
from src.training.fasterrcnn.engine import calculate_tp_fn_fp
from src.training.fasterrcnn.data import (
                                        load_dataloader
                                        )
from src.training.fasterrcnn.patchify import Patchify
from typing import Dict, Any
from sahi import AutoDetectionModel 

def load_trained_fasterrcnn_model(custom_model_args:Dict[str, Any], num_classes:int, device:torch.device, model_dir:str, k:int):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, **custom_model_args)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    model.eval()

    model_state_dict = torch.load(f"{model_dir}/fold_{k}.pt", map_location=device)
    model.load_state_dict(model_state_dict)
    return model

def load_trained_yolo_model(custom_model_args:Dict[str, Any], num_classes:int, device:torch.device, model_dir:str, k:int):
    """
    WORK IN PROGRESS
    """
    model_path = f"{model_dir}/weights/best.pt"
    detection_model = AutoDetectionModel.from_pretrained(
                                                        model_type="yolov8",
                                                        model_path=model_path,
                                                        confidence_threshold=0.3,
                                                        device=device
                                                        )
    return detection_model

def fasterrcnn_predict(model, patch_images, coords, device):
    outputs = model(patch_images)
    outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
    for x in range(len(outputs)):
        # Add starting co-ordinates to the boxes
        outputs[x]["boxes"][:, 0] += coords[x][0]
        outputs[x]["boxes"][:, 1] += coords[x][1]
        outputs[x]["boxes"][:, 2] += coords[x][0]
        outputs[x]["boxes"][:, 3] += coords[x][1]
    return outputs

def k_fold_testing(
        test_dl:torch.utils.data.DataLoader, 
        model_loading_function:callable, 
        model_predict_function:callable,
        model_loading_args, 
        model_dir:str, 
        device:torch.device,
        num_classes:int, 
        patch_size:int,
        config_params:Dict[str, Any]
        ):

    
    k_fold_results = {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0, "mean_iou": 0.0, "mean_precision": 0.0, "mean_recall": 0.0, "f1_score": 0.0}
    patchify = Patchify()
    for k in range(1, 6):

        model = model_loading_function(custom_model_args=model_loading_args, num_classes=num_classes, device=device, model_dir=model_dir, k=k)

        image_results = {k: 0.0 for k in k_fold_results.keys()}
        for i, (images, target) in enumerate(test_dl):
            images = [img.to(device) for img in images]
            target = [{k: v.cpu() for k, v in t.items()} for t in target]

            full_image = images[0] # Only one image in the batch, unpack it
            full_target = target[0] # Only one target in the batch, unpack it

            # Get the patches
            patches, coords = patchify(full_image, patch_size=(patch_size, patch_size))

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
                    outputs = model_predict_function(model, patch_images, coords, device)

                    # Append the all_preds
                    for x in range(len(outputs)):
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