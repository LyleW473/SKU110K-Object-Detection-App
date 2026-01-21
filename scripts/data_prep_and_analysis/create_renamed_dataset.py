import set_path
import os
import pandas as pd
import shutil

if __name__ == "__main__":
    annotations_dir = "data/annotations"
    images_dir = "data/filtered_images"

    # Load the annotations
    columns = ["image_name", "x1", "y1", "x2", "y2", "class", "image_width", "image_height"] # Columns from README
    train_annotations = pd.read_csv(os.path.join(annotations_dir, "annotations_train.csv"), header=None)
    val_annotations = pd.read_csv(os.path.join(annotations_dir, "annotations_val.csv"), header=None)
    test_annotations = pd.read_csv(os.path.join(annotations_dir, "annotations_test.csv"), header=None)
    train_annotations.columns = columns
    val_annotations.columns = columns
    test_annotations.columns = columns
    
    # Combine all annotations into a single dataframe
    all_annotations = pd.concat([train_annotations, val_annotations, test_annotations], ignore_index=True)

    print(train_annotations)
    print(val_annotations)
    print(test_annotations)
    print(all_annotations)
    assert len(all_annotations) == (len(train_annotations) + len(val_annotations) + len(test_annotations)), "Number of rows not equal."

    # Rename the images and update the annotations
    all_images = os.listdir(images_dir)
    new_data = {column: [] for column in columns}
    new_images_dir = "data/renamed_dataset"
    os.makedirs("data/renamed_dataset")
    
    for i in range(len(all_images)):
        print(f"Processing image {i+1}/{len(all_images)} | Progress: {round((i+1)/len(all_images)*100, 4)}%")
        old_image_name = all_images[i]
        new_image_name = f"image_{str(i)}" + ".jpg"
        old_image_path = os.path.join(images_dir, old_image_name) # e.g., data/images/val_95.jpg
        new_image_path= os.path.join(new_images_dir, new_image_name) # e.g., data/fasterrcnn_dataset/image_{i}.jpg
        # print(old_image_name, new_image_name)

        # Find all corresponding annotations with the old image name
        # print(all_annotations.loc[all_annotations["image_name"] == all_images[i]])
        all_annotations.loc[all_annotations["image_name"] == all_images[i], "image_name"] = new_image_name
        # print(all_annotations.loc[all_annotations["image_name"] == new_image_name])

        # Copy the image to the new directory
        shutil.copy(old_image_path, new_image_path)

    # Save the new annotations (with the columns and new image names)
    all_annotations.to_csv(os.path.join(annotations_dir, "annotations_all.csv"), index=False)