import gradio as gr
import numpy as np
import torch

from PIL import ImageDraw, ImageEnhance, ImageFilter, Image
from PIL.Image import Image as PILImage
from typing import Union, List, Tuple

def preprocess_image(
                    image:np.ndarray,
                    blur_amount:float,
                    brightness_level:float, 
                    rotation_angle:float
                    ) -> PILImage:
    """
    Preprocesses the input image by applying brightness adjustment, blur, and rotation.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        blur_amount (float): The amount of blur to apply.
        brightness_level (float): The brightness adjustment level.
        rotation_angle (float): The rotation angle in degrees.
    """
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Apply brightness adjustment
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_level)
    
    # Apply blur
    if blur_amount > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_amount))
    
    # Apply rotation with white background
    if rotation_angle != 0:
        # Convert to RGBA to handle transparency
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        
        rotated = image.rotate(rotation_angle, expand=True, fillcolor=(255, 255, 255, 255))
        
        # Convert back to RGB
        image = Image.new('RGB', rotated.size, (255, 255, 255))
        image.paste(rotated, mask=rotated.split()[3]) 
    
    return image

def preview_update(
                image:Union[np.ndarray, None],
                blur_amount:float,
                brightness_level:float,
                rotation_angle:float
                ) -> Union[PILImage, None]:
    """
    Previews the image with the specified adjustments.

    Args:
        image (Union[np.ndarray, None]): The input image as a NumPy array.
        blur_amount (float): The amount of blur to apply.
        brightness_level (float): The brightness adjustment level.
        rotation_angle (float): The rotation angle in degrees.
    """
    if image is None:
        return None
    return preprocess_image(image, blur_amount, brightness_level, rotation_angle)

def detect_objects(
                model:torch.nn.Module,
                image:Union[np.ndarray, None],
                confidence:float,
                blur_amount:float,
                brightness_level:float,
                rotation_angle:float
                ) -> Union[PILImage, None]:
    """
    Core function to detect objects in the input image.

    Args:
        image (Union[np.ndarray, None]): The input image as a NumPy array.
        confidence (float): The confidence threshold for object detection.
        blur_amount (float): The amount of blur to apply.
        brightness_level (float): The brightness adjustment level.
        rotation_angle (float): The rotation angle in degrees.
    """
    if image is None:
        return None, None
        
    # First preprocess the image
    processed_image = preprocess_image(image, blur_amount, brightness_level, rotation_angle)
    
    # Perform object detection
    results = model(processed_image, conf=confidence)
    
    # Draw bounding boxes
    img = processed_image.copy()
    draw = ImageDraw.Draw(img)
    
    # Prepare detection data for table
    detection_data = []
    for result in results:
        for box in result.boxes:
            xmin, ymin, xmax, ymax = [int(val) for val in box.xyxy[0]]
            confidence_score = float(box.conf[0])
            # Draw rectangle with thicker border
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='green', width=8)
            draw.text((xmin, ymin-20), f'{confidence_score:.2f}', fill='green', font_size=20)
            
            # Add to detection data
            detection_data.append([
                f"{confidence_score:.2f}",
                f"({xmin}, {ymin})",
                f"({xmax}, {ymax})"
            ])
    
    # If no detections, return empty list for table to render properly
    if not detection_data:
        detection_data = []
    
    return img, detection_data

def preview_multiple(
                    files:List[gr.File],
                    blur_amount:float,
                    brightness_level:float,
                    rotation_angle:float
                    ) -> Union[List[PILImage], None]:
    """
    Generate previews for multiple images.

    Args:
        files (List[gr.File]): A list of the uploaded image files.
        blur_amount (float): The amount of blur to apply.
        brightness_level (float): The brightness adjustment level.
        rotation_angle (float): The rotation angle in degrees.
    """
    if not files:
        return None
    
    previews = []
    for file in files:
        img = Image.open(file.name)
        preview = preview_update(img, blur_amount, brightness_level, rotation_angle)
        if preview is not None:
            previews.append(preview)
    
    return previews

class ImageBatchProcessor:
    """
    Class for processing multiple images for object detection.

    - Is used because the model itself cannot be passed to the
      detection button click handler.
    """
    def __init__(self, model:torch.nn.Module):
        """
        Initialises a ImageBatchProcessor object with the specified model.

        Args:
            model (torch.nn.Module): The model to use for object detection.
        """
        self.model = model
    
    def process_multiple_images(
                            self,
                            images:Union[List[gr.File], None],
                            confidence:float,
                            blur_amount:float,
                            brightness_level:float,
                            rotation_angle:float
                            ) -> Union[Tuple[None, None], Tuple[List[PILImage], List[List[str]]]]:
        """
        Process multiple images for object detection.

        Args:
            images (Union[List[gr.File], None]): The list of uploaded images.
            confidence (float): The confidence threshold for object detection.
            blur_amount (float): The amount of blur to apply.
            brightness_level (float): The brightness adjustment level.
            rotation_angle (float): The rotation angle in degrees.
        """
        if not images:
            return None, None
        
        all_results = []
        all_data = []
        
        for idx, file in enumerate(images):
            # Open image from file
            img = Image.open(file.name)
            result, data = detect_objects(self.model, img, confidence, blur_amount, brightness_level, rotation_angle)
            if result is not None:
                all_results.append(result)
                # Add image number to each detection in data
                image_data = [[f"Image {idx+1}"] + row for row in data]
                all_data.extend(image_data)
        
        return all_results, all_data
