# **Installation steps** 
### **1. Create a virtual environment.**  
python -m venv .venv

### **2. Activate the virtual environment**  
.venv/scripts/activate

### **3. Run the command to install all of the requirements:** 
pip install -r requirements.txt

### **4. (OPTIONAL) If training the models or if you wish to run the models on GPUs, download the GPU version from PyTorch, for example:**  
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# **System requirements** 
The requirements were generated using 'pip freeze' for compatability with training, development and deployment, use Python 3.11.9 version.

# **Dataset preparation**

## For YOLO models:
1. Download the dataset and place it the images and annotations directories inside of a 'data' directory, e.g., 'data/images' and 'data/annotations'.
2. Run the 'filter_corrupted_images.py' script.
3. Run the 'create_renamed_dataset.py' script.

### If training on patch images:  
5. Run the 'create_patch_dataset.py' script using the '640' patch size.
6. Run the 'custom_yolo_dataset_splitting.py' script using the 'patches_640' option.
7. Run the 'create_yolo_dataset.py' script using the 'patches_640' option
### CLEARML DATASET ID: 4a4e1133282647328a0a32fc1c6755a4

### If training on the complete images (Resized):  
4. Run the 'custom_yolo_dataset_splitting.py' script using 'normal' option.
5. Run the 'create_yolo_dataset.py' script using the 'custom' option.

### CLEARML DATASET ID: c9af86f6ec274d39bbb52b5817098081

# **Training + Inference**
1. Run the 'train_yolo.py' script, using either the 'custom' or 'patches_640' option, depending on what you chose in the dataset preparation.
2. Running step 1 will only find the optimal hyperparameters, run the script again to train the final model on the generated hyperparameters.
3. Run 'inference.ipynb', referencing the model, to see predictions on a single test image.
4. Run 'test_yolo.py' to see predictions on an entire directory of test images.

### CLEARML MODEL ID: 63bf2170fc084c69ae63bde007705b8c

# **Hugging Face Spaces**
Deployed app here: https://huggingface.co/spaces/muratt0/YOLO_Object_Detection

