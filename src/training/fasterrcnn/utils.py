import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

from PIL import Image
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple
from clearml import Task

def load_environment_variables(env_path:str) -> None:
    """
    Loads the environment variables from the .env file
    required for training using ClearML.

    Args:
        env_path (str): Path to the '.env' file containing the environment variables.
    """
    
    # Load environment variables from .env file
    load_dotenv(dotenv_path=env_path)

    try:
        os.environ["CLEARML_WEB_HOST"] = os.getenv("CLEARML_WEB_HOST")
        os.environ["CLEARML_API_HOST"] = os.getenv("CLEARML_API_HOST")
        os.environ["CLEARML_FILES_HOST"] = os.getenv("CLEARML_FILES_HOST")
        os.environ["CLEARML_API_ACCESS_KEY"] = os.getenv("CLEARML_API_ACCESS_KEY")
        os.environ["CLEARML_API_SECRET_KEY"] = os.getenv("CLEARML_API_SECRET_KEY")

        # Check if all variables were loaded successfully
        if None in (os.getenv("CLEARML_WEB_HOST"), os.getenv("CLEARML_API_HOST"), os.getenv("CLEARML_FILES_HOST"),
                    os.getenv("CLEARML_API_ACCESS_KEY"), os.getenv("CLEARML_API_SECRET_KEY")):
            raise ValueError("Please ensure the .env file contains all required environment variables: CLEARML_WEB_HOST, CLEARML_API_HOST, CLEARML_FILES_HOST, CLEARML_API_ACCESS_KEY, CLEARML_API_SECRET_KEY")
    
    except Exception as e:
        raise ValueError("Failed to load environment variables. Please check the .env file.") from e
    
def initialise_clearml_task(configuration_params:Dict[str, Any], project_name="Object Detection Project", task_name="Cross Validation Training") -> Task:
    """
    Initialises a ClearML task for logging and monitoring the training 
    process and connects to it with the configuration parameters.

    Args:
        configuration_params (Dict[str, Any]): Dictionary containing the configuration parameters for the task.
    """
    task = Task.init(project_name=project_name, 
                    task_name=task_name,
                    output_uri=True)

    task.connect(configuration_params)
    return task

def get_image_paths(directory:str) -> List[str]:
    """
    Returns a list of file paths to images in a specified directory.

    Args:
        directory (str): The directory containing the images.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory)]

def calculate_dataset_statistics(image_paths:List[str], chosen_option:str) -> Tuple[List[float], List[float]]:
    """
    Calculates the mean and standard deviation of the dataset and
    saves it as a JSON file.

    Args:
        image_paths (List[str]): A list of image file paths used to calculate the statistics.
        chosen_option (str): The dataset option chosen for calculating the statistics.
    """
    if os.path.exists(f"dataset_statistics/stats_{chosen_option}.json"):
        print("Dataset statistics already calculated.")
        with open(f"dataset_statistics/stats_{chosen_option}.json", "r") as f:
            stats = json.load(f)
        return stats["mean"], stats["std"]

    num_channels = 3
    channel_sums = np.zeros(num_channels)
    channel_squared_sums = np.zeros(num_channels)
    channel_maxs = np.ones(num_channels) * -np.inf
    channel_mins = np.ones(num_channels) * np.inf
    num_pixels = 0

    num_images = len(image_paths)
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i + 1}/{num_images} | Progress: {((i + 1) / num_images) * 100:.5f}%")
        image = Image.open(image_path)
        image = np.array(image)
        image = image / 255.0 # Normalise to [0, 1] first (The normalisation will be applied to [0, 1] images)
        image = np.transpose(image, (2, 0, 1)) # [H, W, C] -> [C, H, W]

        channel_sums += np.sum(image, axis=(1, 2))
        channel_squared_sums += np.sum(image ** 2, axis=(1, 2))
        num_pixels += image.shape[1] * image.shape[2] # H * W

        channel_maxs = np.maximum(channel_maxs, np.max(image, axis=(1, 2)))
        channel_mins = np.minimum(channel_mins, np.min(image, axis=(1, 2)))

    channel_means = channel_sums / num_pixels
    channel_std_devs = np.sqrt((channel_squared_sums / num_pixels) - channel_means ** 2)

    # Save the statistics
    os.makedirs("dataset_statistics", exist_ok=True)
    means = channel_means.tolist()
    std_devs = channel_std_devs.tolist()
    with open(f"dataset_statistics/stats_{chosen_option}.json", "w") as f:
        json.dump({"mean": means, "std": std_devs}, f)
    return means, std_devs

def save_image(image:torch.Tensor, target:Dict[str, torch.Tensor], name:str) -> None:
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())
    for box in target["boxes"]:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    # Save image
    plt.savefig(name)
    plt.close()