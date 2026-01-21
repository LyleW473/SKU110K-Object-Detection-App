import copy
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to prevent plots from showing up
import matplotlib.pyplot as plt
import torch

from clearml import Logger
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple

def train_model_k_folds(
                        model:torch.nn.Module,
                        train_dataloaders:Dict[str, DataLoader],
                        val_dataloaders:Dict[str, DataLoader],
                        device:torch.device,
                        config_params:Dict[str, Any],
                        model_dir:str,
                        print_interval:int
                        ) -> None:
    """
    Trains a Faster R-CNN model using K-Fold cross-validation 
    with the provided training and validation data loaders for
    each fold and the specified configuration parameters.

    Args:
        model (torch.nn.Module): The Faster R-CNN model to train.
        train_dataloaders (Dict[str, DataLoader]): The data loaders for the training data for each fold.
        val_dataloaders (Dict[str, DataLoader]): The data loaders for the validation data for each fold.
        device (torch.device): The device to run the training on (CPU or GPU).
        config_params (Dict[str, Any]): The configuration parameters for training.
        model_dir (str): The directory to save the best model for each fold.
        print_interval (int): The interval to print out metrics during training.
    """
    for i in range(5):
        fold = (i + 1)  

        train_loader = train_dataloaders[f'fold_{fold}_train']
        val_loader = val_dataloaders[f'fold_{fold}_val']
    
        best_model = train_model(
                                model=model,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                device=device,
                                config_params=config_params,
                                print_interval=print_interval
                                )
        
        fold_model_path = f"{model_dir}/fold_{fold}.pt"
        torch.save(best_model.state_dict(), fold_model_path)

    # TODO: Currently returns the best model for the last fold only
    return best_model, fold_model_path
    
def forward_pass_loss(
                    model:torch.nn.Module,
                    optimiser:torch.optim.Optimizer,
                    data_loader:torch.utils.data.DataLoader,
                    device:torch.device,
                    logger:Logger,
                    epoch:int,
                    n_epochs:int,
                    mode:str,
                    print_interval:int
                    ) -> float:
    """
    The forward pass for the train/validation phases to retrieve the 
    average loss for the epoch.

    Args:
        model (torch.nn.Module): The Faster R-CNN model to train/validate.
        optimiser (torch.optim.Optimizer): The optimizer to use for training/validation.
        data_loader (torch.utils.data.DataLoader): The data loader for the training/validation data.
        device (torch.device): The device to run the training on (CPU or GPU).
        logger (Logger): The ClearML logger to log the loss values.
        epoch (int): The current epoch number.
        n_epochs (int): The total number of epochs.
        mode (str): The mode of the forward pass (train/val).
        print_interval (int): The interval to print out metrics during training.
    """

    epoch_loss = 0.0
    torch.cuda.empty_cache() # Clear GPU cache to prevent memory leaks

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Zero-grad if training
        if mode == "train":
            optimiser.zero_grad()
        
        # Calculate batch loss
        loss_dict = model(images, targets)
        batch_loss = sum(loss for loss in loss_dict.values())

        # Backpropagate if training
        if mode == "train":
            batch_loss.backward()
            optimiser.step()
        
        # Convert batch loss to CPU and get the value
        batch_loss = batch_loss.to("cpu").item()
        epoch_loss += batch_loss
        avg_loss = epoch_loss / (i + 1)
    
        with torch.no_grad():
            # Log batch loss to clearml
            logger.report_scalar(
                                title=f'{mode} loss - loss per batch', 
                                series='loss',
                                value=batch_loss,
                                iteration=epoch * len(data_loader) + i
                                )

            # Log running loss to clearml
            logger.report_scalar(
                                title=f'{mode} loss - running loss',
                                series='loss',
                                value=avg_loss,
                                iteration=epoch * len(data_loader) + i
                                )

        if (i % print_interval == 0) or (i == len(data_loader) - 1):
            print(f"{mode} | Epoch: {epoch + 1}/{n_epochs} | Batch: {i + 1}/{len(data_loader)} | Batch loss: {batch_loss} | AvgLoss: {avg_loss:.5f}")

    avg_loss = epoch_loss / len(data_loader)
    print(f'Epoch {epoch + 1}. {mode} epoch complete with average loss: {avg_loss}')
    logger.report_scalar(title=f'{mode} loss - loss per epoch', series='loss', value=avg_loss, iteration=epoch)
    return avg_loss

def calculate_tp_fn_fp(
                    iou:torch.Tensor, 
                    pred_boxes:torch.Tensor,
                    pred_labels:torch.Tensor,
                    target_boxes:torch.Tensor,
                    target_labels:torch.Tensor,
                    iou_threshold:float=0.5
                    ):
    """
    Calculates the True Positives, False Negatives, and False Positives for a batch of predictions

    Args:
        iou (torch.Tensor): The IoU matrix for the batch of predictions and targets.
        pred_boxes (torch.Tensor): The predicted bounding boxes for the batch.
        pred_labels (torch.Tensor): The predicted labels for the batch.
        target_boxes (torch.Tensor): The target bounding boxes for the batch.
        target_labels (torch.Tensor): The target labels for the batch.
        iou_threshold (float): The IoU threshold for matching predictions to targets.
    """
    tp, fn, fp = 0, 0, 0
    matched_gt = torch.zeros(len(target_boxes), dtype=torch.bool)
    matched_pred = torch.zeros(len(pred_boxes), dtype=torch.bool)

    for gt_idx in range(0, len(target_boxes)):
        if not matched_gt[gt_idx]: # Have not matched this GT box yet
            # Get the max IoU and the index of the best prediction
            max_iou, best_pred_idx = iou[:, gt_idx].max(0) 

            # Best prediction greater than threshold and not matched yet
            if (max_iou > iou_threshold and 
                not matched_pred[best_pred_idx] and
                pred_labels[best_pred_idx] == target_labels[gt_idx]):
                tp += 1
                matched_gt[gt_idx] = True
                matched_pred[best_pred_idx] = True
            else:
                fn += 1
    inverted_matched_pred = ~matched_pred # Invert False to True and True to False
    fp += (inverted_matched_pred).sum().item() # All of the unmatched predictions are false positives
    return tp, fn, fp

def forward_pass_evaluation_metrics(
                                    model:torch.nn.Module,
                                    data_loader:torch.utils.data.DataLoader,
                                    device:torch.device,
                                    logger:Logger,
                                    epoch:int,
                                    mode:str,
                                    print_interval:int
                                    ) -> Dict[str, Any]:
    """
    The forward pass for the validation or testing phase to calculate:
    - mAP (Mean Average Precision)
    - mAP(50) and map(75)
    - IoU (Intersection over Union)
    - Precision, Recall, and F1 Score
    for the model over one epoch.

    Args:
        model (torch.nn.Module): The Faster R-CNN model to evaluate.
        data_loader (torch.utils.data.DataLoader): The data loader for the validation/test data.
        device (torch.device): The device to run the evaluation on (CPU or GPU).
        logger (Logger): The ClearML logger to log the IoU values.
        epoch (int): The current epoch number.
        mode (str): The mode of the forward pass (val/test).
        print_interval (int): The interval to print out metrics during training.
    """

    map_metric = MeanAveragePrecision()
    total_iou = 0
    total_precision = 0
    total_recall = 0
    iou_threshold = 0.5 # IoU threshold for precision and recall matching
    for i, (images, targets) in enumerate(data_loader):

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions = model(images)  # Get predictions without targets

        # Update mAP metric
        map_metric.update(preds=predictions, target=targets)

        # Calculate the IoU for the current batch and log it
        batch_iou = 0.0

        tp, fp, fn = 0, 0, 0
        for pred, target in zip(predictions, targets):
            pred_boxes, pred_labels = pred["boxes"], pred["labels"]
            target_boxes, target_labels = target["boxes"], target["labels"]

            if len(pred_boxes) == 0 or len(target_boxes) == 0:
                fn += len(target_boxes) # Meaning that we missed all the GT boxes
                continue

            # IoU, Precision + Recall
            iou = box_iou(pred_boxes, target_boxes)
            additive_tp, additive_fn, additive_fp = calculate_tp_fn_fp(
                                                                        iou=iou,
                                                                        pred_boxes=pred_boxes,
                                                                        pred_labels=pred_labels,
                                                                        target_boxes=target_boxes,
                                                                        target_labels=target_labels,
                                                                        iou_threshold=iou_threshold
                                                                        )
            tp += additive_tp
            fn += additive_fn
            fp += additive_fp

            # Mean batch IoU
            mean_iou = iou.mean().item()
            batch_iou += mean_iou
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        total_precision += precision
        total_recall += recall

        batch_iou /= len(targets)
        total_iou += batch_iou
        avg_iou = total_iou / (i + 1) # Divide by number of batches to get average IoU


        if (i % print_interval == 0) or (i == len(data_loader) - 1):
            print(f"{mode} | Batch: {i + 1}/{len(data_loader)} | Batch IoU: {batch_iou:.5f} | AvgIoU: {avg_iou:.5f} | Precision: {precision:.5f} | Recall: {recall:.5f} | F1 Score: {f1_score} | TP: {tp} | FP: {fp} | FN: {fn}")

        # Log the IoU, precision and recall for the batch
        logger.report_scalar(
                            title=f"{mode.capitalize()} batch IoU",
                            series="iou",
                            value=mean_iou,
                            iteration=epoch * len(data_loader) + i
                            )
        logger.report_scalar(
                            title=f"{mode.capitalize()} batch Precision",
                            series="precision",
                            value=precision,
                            iteration=epoch * len(data_loader) + i
                            )  
        logger.report_scalar(
                            title=f"{mode.capitalize()} batch Recall",
                            series="recall",
                            value=recall,
                            iteration=epoch * len(data_loader) + i
                            )
        logger.report_scalar(
                            title=f"{mode.capitalize()} batch F1 Score",
                            series="f1_score",
                            value=f1_score,
                            iteration=epoch * len(data_loader) + i
                            )
    num_batches = len(data_loader)
    avg_iou = total_iou / num_batches # Calculate average IoU for the epoch
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches
    final_f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
            
    # Calculate the mAP results (mAP, mAP(50), mAP(75))
    mAP_results = map_metric.compute()

    results = {
                "map_results": mAP_results, 
                "avg_iou": avg_iou, 
                "avg_precision": avg_precision, 
                "avg_recall": avg_recall, 
                "f1_score": final_f1_score
                }
    return results

def train_model(
                model:torch.nn.Module,
                train_loader:torch.utils.data.DataLoader,
                val_loader:torch.utils.data.DataLoader,
                device:torch.device,
                config_params:Dict[str, Any],
                print_interval:int
                ) -> torch.nn.Module:
    """
    Trains a Faster R-CNN model with the provided training + validation data
    loaders and the specified configuration parameters.

    Args:
        model (torch.nn.Module): The Faster R-CNN model to train.
        train_loader (torch.utils.data.DataLoader): The data loader for the training data.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation data.
        device (torch.device): The device to run the training on (CPU or GPU).
        config_params (Dict[str, Any]): The configuration parameters for training.
        print_interval (int): The interval to print out metrics during training.
    """
    n_epochs = config_params['num-epochs']  
    optimizer = torch.optim.SGD(
                                model.parameters(), 
                                momentum=config_params['momentum'],
                                lr=config_params['learning-rate'],
                                )

    best_val_loss = float("inf")
    best_model = copy.deepcopy(model)
    
    # clearml logger
    logger = Logger.current_logger()

    avg_train_epoch_losses = []
    avg_val_epoch_losses = []

    # Start training
    for epoch in range(n_epochs):

        # Train phase
        model.train()  
        avg_train_epoch_loss = forward_pass_loss(
                                            model=model,
                                            optimiser=optimizer,
                                            data_loader=train_loader,
                                            device=device,
                                            logger=logger,
                                            epoch=epoch,
                                            n_epochs=n_epochs,
                                            mode="train",
                                            print_interval=print_interval
                                            )
        avg_train_epoch_losses.append(avg_train_epoch_loss)

        # Validation phase

        # Step 1: Calculate validation loss
        model.train()  # Temporarily set to train mode to calculate loss
        with torch.no_grad():
            avg_val_epoch_loss = forward_pass_loss(
                                                model=model,
                                                optimiser=optimizer,
                                                data_loader=val_loader,
                                                device=device,
                                                logger=logger,
                                                epoch=epoch,
                                                n_epochs=n_epochs,
                                                mode="val",
                                                print_interval=print_interval
                                                )
            avg_val_epoch_losses.append(avg_val_epoch_loss)

        # Step 2: Generate predictions for mAP and IoU calculation
        model.eval()  # Set back to eval mode to get predictions
        with torch.no_grad():
            results = forward_pass_evaluation_metrics(
                                                    model=model,
                                                    data_loader=val_loader,
                                                    device=device,
                                                    logger=logger,
                                                    epoch=epoch,
                                                    mode="val",
                                                    print_interval=print_interval
                                                    )
            mAP_results = results["map_results"]
            avg_iou = results["avg_iou"]
            avg_precision = results["avg_precision"]
            avg_recall = results["avg_recall"]
            f1_score = results["f1_score"]
            
            logger.report_scalar(title="Mean Average Precision (mAP)", series="mAP", value=mAP_results["map"], iteration=epoch)
            logger.report_scalar(title="mAP(50)", series="mAP_50", value=mAP_results["map_50"], iteration=epoch)
            logger.report_scalar(title="mAP(75)", series="mAP_75", value=mAP_results["map_75"], iteration=epoch)
            logger.report_scalar(title="Mean IoU (epochs)", series="IoU", value=avg_iou, iteration=epoch)
            logger.report_scalar(title="Mean Precision (epochs)", series="Precision", value=avg_precision, iteration=epoch)
            logger.report_scalar(title="Mean Recall (epochs)", series="Recall", value=avg_recall, iteration=epoch)
            logger.report_scalar(title="F1 Score (epochs)", series="F1 Score", value=f1_score, iteration=epoch)

            # plot the line chart comparing training loss and validation loss
            plt.figure()
            plt.plot(range(1, epoch + 2), avg_train_epoch_losses, label="Training Loss", color="blue")
            plt.plot(range(1, epoch + 2), avg_val_epoch_losses, label="Validation Loss", color="orange")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training vs Validation Loss")
            plt.legend()
            
            # upload the chart into clearml
            Logger.current_logger().report_matplotlib_figure(
                title="Training vs Validation Loss",
                series="Loss Comparison",
                figure=plt,
                iteration=epoch
            )
            plt.close()
        
            print(f"Val | mAP {mAP_results['map']} | mAP_50: {mAP_results['map_50']} | mAP_75: {mAP_results['map_75']} | Mean IoU: {avg_iou} | Mean Precision: {avg_precision} | Mean Recall: {avg_recall}")

            # Check for best model
            if avg_val_epoch_loss < best_val_loss:
                best_val_loss = avg_val_epoch_loss
                best_model = copy.deepcopy(model)
                print(f'Epoch {epoch + 1}, new best model found with validation loss: {best_val_loss}')

    # # save the best model
    # best_model_path = prefix + 'best_model.pt'
    # torch.save(best_model.state_dict(), best_model_path)
    # print(f"Best model saved at: {best_model_path}")
    logger.report_single_value('Best validation loss', best_val_loss)
    return best_model

def evaluate_on_test_set(
                        model:torch.nn.Module,
                        test_loader:torch.utils.data.DataLoader,
                        device:torch.device,
                        ) -> Dict[str, float]:
    """
    Evaluates a Faster R-CNN model on the test set using the provided test data loader.

    Args:
        model (torch.nn.Module): The Faster R-CNN model to evaluate.
        test_loader (torch.utils.data.DataLoader): The data loader for the test data.
        device (torch.device): The device to run the evaluation on (CPU or GPU).
    """
    model.eval()  # Set model to evaluation mode
    map_metric = MeanAveragePrecision()
    logger = Logger.current_logger()  # Get ClearML logger

    with torch.no_grad():
        results = forward_pass_evaluation_metrics( 
                                                model=model,
                                                data_loader=test_loader,
                                                device=device,
                                                logger=logger,
                                                epoch=0,
                                                mode="test",
                                                print_interval=1
                                                )
        mAP_results = results["map_results"]
        avg_iou = results["avg_iou"]
        avg_precision = results["avg_precision"]
        avg_recall = results["avg_recall"]
        f1_score = results["f1_score"]

        
        print("Evaluation Results on Test Set:")
        print(f"Mean Average Precision (mAP): {mAP_results['map']}")
        print(f"mAP(50): {mAP_results['map_50']}")
        print(f"mAP(75): {mAP_results['map_75']}")
        print(f"Mean IoU: {avg_iou}")
        print(f"Mean Precision: {avg_precision}")
        print(f"Mean Recall: {avg_recall}")
        print(f"F1 Score: {f1_score}")

        # Log final mAP, mAP(50), mAP(75), and IoU results to ClearML
        logger.report_single_value("Test Set Mean Average Precision (mAP)", mAP_results["map"])
        logger.report_single_value("Test Set mAP(50)", mAP_results["map_50"])
        logger.report_single_value("Test Set mAP(75)", mAP_results["map_75"])
        logger.report_single_value("Test Set Mean IoU", avg_iou)
        logger.report_single_value("Test Set Mean Precision", avg_precision)
        logger.report_single_value("Test Set Mean Recall", avg_recall)
        logger.report_single_value("Test Set F1 Score", f1_score)

        # Clear the metric for the next evaluation
        map_metric.reset()

    # Return the evaluation results
    results = {
                "mAP": mAP_results["map"], 
                "mAP_50": mAP_results["map_50"], 
                "mAP_75": mAP_results["map_75"], 
                "mean_iou": avg_iou,
                "mean_precision": avg_precision,
                "mean_recall": avg_recall,
                "f1_score": f1_score
                }
    return results