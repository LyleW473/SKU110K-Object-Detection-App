import set_path
import os
import cv2

if __name__ == "__main__":
    images_dir = "data/images"

    all_images = os.listdir(images_dir)
    new_location = "data/filtered_images"
    os.makedirs(new_location)

    # Re-write all the images to get rid of any corrupted images
    for i, image_name in enumerate(all_images):
        image_name = all_images[i]
        image_path = f"{images_dir}/{image_name}"
        print(f"Processing image {i+1}/{len(all_images)} | Progress: {round((i+1)/len(all_images)*100, 4)}%", image_path)

        image = cv2.imread(image_path)
        cv2.imwrite(f"{new_location}/{image_name}", image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])