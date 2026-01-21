import albumentations as A

def construct_image_transforms(chosen_option:str) -> A.Compose:
    """
    Construct the image transforms to be applied to the images in the dataset.
    - Convert the PIL Image to a Tensor.
    - Ensure that the image is of type uint8.
    - Applies data augmentation transforms.
    - Normalise the image to be in the range [0, 1].
    """
    if chosen_option.startswith("patch"):
        return A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5), # (0 or more times)
                    A.ChannelShuffle(p=0.25),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
                    )
    else:
        return A.Compose([
                    A.RandomSizedBBoxSafeCrop(height=512, width=512, erosion_rate=0.0, interpolation=1, p=1.0), # Always apply
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5), # (0 or more times)
                    A.ChannelShuffle(p=0.25),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
                    )