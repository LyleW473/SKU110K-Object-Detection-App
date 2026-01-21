import pandas as pd
import numpy as np
import albumentations

from typing import Tuple, List, Dict, Callable, Union
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from src.training.fasterrcnn.dataset import CustomDataset

def load_annotations(annotations_path:str) -> pd.DataFrame:
    """
    Loads the annotations from a CSV file.

    Args:
        annotations_path (str): Path to the CSV file containing the annotations.

    Returns:
        pd.DataFrame: Dataframe containing the annotations.
    """

    annotations_df = pd.read_csv(annotations_path)
    return annotations_df

def create_processed_dataframes(annotations_df:pd.DataFrame, num_objects_bins=[0, 2, 5, 10, np.inf], use_categories=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes the annotations dataframe and creates a new dataframe with image statistics for stratification.

    Args:
        annotations_df (pd.DataFrame): Dataframe containing annotations for the images.
        num_objects_bins (List[int]): List of integers to use as bins for the number of objects.
        use_categories (bool): Whether to use categories for stratification.
    """

    annotations_df["bbox_area"] = (annotations_df["x2"] - annotations_df["x1"]) * (annotations_df["y2"] - annotations_df["y1"])

    # Group by image name and calculate the number of objects and the average bbox area for each image
    image_statistics = annotations_df.groupby("image_name").agg(
                                                                num_objects=("image_name", "size"),
                                                                avg_bbox_area=("bbox_area", "mean")
                                                                ).reset_index()
    
    # Create a combined feature used for stratification
    if not use_categories:
        image_statistics["combined_feature"] = image_statistics["num_objects"].astype(str) + "_" + image_statistics["avg_bbox_area"].astype(str)
    else:
        image_statistics["num_objects_category"] = pd.cut(
                                                    image_statistics["num_objects"], 
                                                    bins=num_objects_bins, 
                                                    labels=["Few", "Moderate", "Above Average", "Many"]
                                                    )
        image_statistics["avg_bbox_area_category"] = pd.qcut(
                                                        image_statistics["avg_bbox_area"], 
                                                        q=4, 
                                                        labels=["Small", "Medium", "Large", "Very Large"]
                                                        )
        image_statistics["combined_feature"] = image_statistics["num_objects_category"].astype(str) + "_" + image_statistics["avg_bbox_area_category"].astype(str)

    # Bin "num_objects" and "avg_bbox_area" features for stratification
    image_statistics["num_objects_bin"] = pd.cut(image_statistics["num_objects"], bins=num_objects_bins, labels=False)
    image_statistics["avg_bbox_area_bin"] = pd.qcut(image_statistics["avg_bbox_area"], q=4, labels=False)

    image_statistics["combined_feature"] = image_statistics["num_objects_bin"].astype(str) + "_" + image_statistics["avg_bbox_area_bin"].astype(str)

    return annotations_df, image_statistics

def train_test_split(
                    annotations_df:pd.DataFrame, 
                    image_statistics:pd.DataFrame,
                    test_size:float=0.5
                    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into training and test sets.

    Args:
        annotations_df (pd.DataFrame): Dataframe containing the annotations for the images.
        image_statistics (pd.DataFrame): Dataframe containing the image statistics.
        test_size (float): Proportion of the data to include in the test split.
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_index, test_index in sss.split(image_statistics, image_statistics["combined_feature"]):
        train_image_names = image_statistics.loc[train_index, "image_name"]
        test_image_names = image_statistics.loc[test_index, "image_name"]

        train_df = annotations_df[annotations_df["image_name"].isin(train_image_names)]
        test_df = annotations_df[annotations_df["image_name"].isin(test_image_names)]

    # Printing the statistics
    total_objects = annotations_df.shape[0]
    train_objects = train_df.shape[0]
    test_objects = test_df.shape[0]
    n_train_images = train_image_names.shape[0]
    n_test_images = test_image_names.shape[0]

    print(f"Total objects: {total_objects}")
    print(f"Train objects: {train_objects} ({train_objects/total_objects*100:.2f}%)")
    print(f"Test objects: {test_objects} ({test_objects/total_objects*100:.2f}%)")
    print()
    print(f"Total images: {image_statistics.shape[0]}")
    print(f"Train images: {n_train_images} ({n_train_images/image_statistics.shape[0]*100:.2f}%)")
    print(f"Test images: {n_test_images} ({n_test_images/image_statistics.shape[0]*100:.2f}%)")

    return train_df, test_df, train_image_names, test_image_names

def create_train_val_folds(
                            annotations_df:pd.DataFrame, 
                            image_statistics:pd.DataFrame, 
                            train_image_names:pd.Series
                            ) -> List[Dict[str, pd.DataFrame]]:
    """
    Uses:
    - The dataframe containing all annotations.
    - The image statistics dataframe containing the number of objects and average bbox area for each image.
    - The series containing the image names for the training set.
    To create training and validation folds.

    Args:
        annotations_df (pd.DataFrame): Dataframe containing the annotations for the images.
        image_statistics (pd.DataFrame): Dataframe containing the image statistics.
        train_image_names (pd.Series): Series containing the image names for the training set.
    """
    # Use only the train image names for generating training and validation folds
    train_image_statistics = image_statistics[image_statistics["image_name"].isin(train_image_names)]

    # Split training set into training and validation folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_splits = []
    for fold_num, (train_index, val_index) in enumerate(skf.split(train_image_statistics, train_image_statistics["combined_feature"])):
        train_image_names = train_image_statistics.iloc[train_index]["image_name"]
        val_image_names = train_image_statistics.iloc[val_index]["image_name"]

        train_df = annotations_df[annotations_df["image_name"].isin(train_image_names)]
        val_df = annotations_df[annotations_df["image_name"].isin(val_image_names)]

        all_splits.append({"train": train_df, "val": val_df})
        
        print(f"Fold {fold_num}")

        n_train_obj = train_df.shape[0]
        n_val_obj = val_df.shape[0]
        total_objects = n_train_obj + n_val_obj
        train_obj_proportion = (n_train_obj / total_objects) * 100
        val_obj_proportion = (n_val_obj / total_objects) * 100
        print(f"Train (objects): {train_df.shape[0]}, Val (objects): {val_df.shape[0]}")
        print(f"Train proportion (objects): {train_obj_proportion:.4f}%, Val proportion (objects): {val_obj_proportion:.4f}%")

        n_train_images = train_image_names.shape[0]
        n_val_images = val_image_names.shape[0]
        total_images = n_train_images + n_val_images
        train_img_proportion = (n_train_images / total_images) * 100
        val_img_proportion = (n_val_images / total_images) * 100
        print(f"Train (images): {n_train_images}, Val (images): {n_val_images}")
        print(f"Train proportion (images): {train_img_proportion:.4f}%, Val proportion (images): {val_img_proportion:.4f}%")
        print()

    return all_splits


def _create_datasets(
                    all_splits:List[Dict[str, pd.DataFrame]], 
                    image_dir:str, 
                    image_transforms:albumentations.Compose
                    ) -> Tuple[Dict[str, CustomDataset], Dict[str, CustomDataset]]:
    """
    Creates two dictionaries of datasets for training and validation 
    across all the folds.

    Args:
        all_splits (List[Dict[str, pd.DataFrame]]): List of dictionaries containing the training and validation dataframes for each fold.
        image_dir (str): Path to the directory containing the images.
        image_transforms (albumentations.Compose): Compose object containing the image transformations to apply.
    """
    train_datasets = {}
    val_datasets = {}
    for i, fold_dict in enumerate(all_splits):
        train_datasets[f'fold_{i+1}_train'] = CustomDataset(fold_dict['train'], image_dir, transforms=image_transforms)
        val_datasets[f'fold_{i+1}_val'] = CustomDataset(fold_dict['val'], image_dir, transforms=image_transforms)
    return train_datasets, val_datasets

def create_train_val_dataloaders(
                                all_splits:List[Dict[str, pd.DataFrame]], 
                                image_dir:str, 
                                image_transforms:albumentations.Compose, 
                                config_params:Dict[str, int],
                                collate_fn:Callable
                                ) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader]]:
    """
    Creates two dictionaries containing the dataloaders for training and validation
    for each fold.

    Args:
        all_splits (List[Dict[str, pd.DataFrame]]): List of dictionaries containing the training and validation dataframes for each fold.
        image_dir (str): Path to the directory containing the images.
        image_transforms (albumentations.Compose): Compose object containing the image transformations to apply.
        config_params (Dict[str, int]): Dictionary containing the configuration parameters.
        collate_fn (Callable): Collate function to use for the DataLoader.
    """

    train_datasets, val_datasets = _create_datasets(all_splits, image_dir, image_transforms)
    
    dataloaders_train = {}
    dataloaders_val = {}

    # # TEMP: Define sample size for testing
    # train_sample_size = 5

    for i in range(5):
        train_key = f'fold_{i+1}_train'
        val_key = f'fold_{i+1}_val'

        # For testing, only take sample size to test on
        # train_subset = Subset(custom_dataset_train[train_key], range(train_sample_size))
        # val_subset = Subset(custom_dataset_val[val_key], range(train_sample_size))

        # dataloaders_train[train_key] = DataLoader(train_subset, batch_size=config_params['batch-size'], shuffle=True, collate_fn=collate_fn)
        # dataloaders_val[val_key] = DataLoader(val_subset, batch_size=config_params['batch-size'], shuffle=True, collate_fn=collate_fn)

        # uncomment for actual training
        dataloaders_train[train_key] = load_dataloader(
                                                    dataset=train_datasets[train_key], 
                                                    config_params=config_params, 
                                                    collate_fn=collate_fn,
                                                    )
        dataloaders_val[val_key] = load_dataloader(
                                                dataset=val_datasets[val_key], 
                                                config_params=config_params, 
                                                collate_fn=collate_fn,
                                                )
        
    return dataloaders_train, dataloaders_val

def load_dataloader(
                    dataset: CustomDataset,
                    config_params:Dict[str, int],
                    collate_fn:Callable,
                    shuffle:bool=True,
                    sample_size:Union[int, None]=None
                    ) -> DataLoader:
    """
    Loads a dataloader for the given annotations and image directory.

    Args:
        dataset (CustomDataset): CustomDataset object containing the annotations and image directory.
        config_params (Dict[str, int]): Dictionary containing the configuration parameters.
        collate_fn (Callable): Collate function to use for the DataLoader.
        shuffle (bool): Whether to shuffle the data.
        sample_size (Union[int, None]): Number of samples to use from the dataset (use if you want 
                                        to use a subset of the data).
    """

    # If you only want a subset for training/testing
    if sample_size is not None:
        dataset = Subset(dataset, range(sample_size))

    dataloader = DataLoader(
                            dataset, 
                            batch_size=config_params['batch-size'], 
                            shuffle=shuffle,
                            collate_fn=collate_fn
                            )
    return dataloader