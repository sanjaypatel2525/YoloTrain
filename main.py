import os
import random
import cv2
import shutil
from ultralytics import YOLO
import kagglehub

basePath="C:/Users/dj525/home-server/Rangoli/"
modelPath= basePath + "model/"

# Paths to folders
dataset_dir = basePath + "dataset/"        # Main dataset directory
dataset_yaml_path = dataset_dir + "rangoli.yml"
input_img_dir = dataset_dir + "allimages/"  # Folder containing all images
train_img_dir = os.path.join(dataset_dir, "images/train")
val_img_dir = os.path.join(dataset_dir, "images/val")
train_lbl_dir = os.path.join(dataset_dir, "labels/train")
val_lbl_dir = os.path.join(dataset_dir, "labels/val")

# Create output directories if they don't exist
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

# Parameters
img_size = (416, 416)  # Target image size for YOLO
train_ratio = 0.8      # 80% for training, 20% for validation
class_id = 0           # Single class ID (e.g., for "rangoli")

# 1. Resize Images, Split into Train/Validation, and Create Placeholder Labels
def prepare_images(input_img_dir, train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir, img_size, train_ratio):
    img_files = [f for f in os.listdir(input_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(img_files)

    train_count = int(len(img_files) * train_ratio)
    train_files = img_files[:train_count]
    val_files = img_files[train_count:]

    for img_name in img_files:
        img_path = os.path.join(input_img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Resize image
        img_resized = cv2.resize(img, img_size)

        # Define output paths
        if img_name in train_files:
            output_img_dir = train_img_dir
            output_lbl_dir = train_lbl_dir
        else:
            output_img_dir = val_img_dir
            output_lbl_dir = val_lbl_dir

        # Save resized image
        cv2.imwrite(os.path.join(output_img_dir, img_name), img_resized)

        # Generate a placeholder label file (mock bounding box in YOLO format)
        img_height, img_width = img_resized.shape[:2]
        x_center, y_center = 0.5, 0.5  # Centered in the image
        width, height = 0.9, 0.9       # 30% of the image width and height

        label_path = os.path.join(output_lbl_dir, img_name.replace('.jpg', '.txt')
                                  .replace('.jpeg', '.txt')
                                  .replace('.png', '.txt'))
        with open(label_path, "w") as label_file:
            label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


#prepare_images(input_img_dir, train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir, img_size, train_ratio)

# 3. Train YOLO Model
model = YOLO(basePath + "model/yolo11n.pt")  # Load YOLOv5 small model

# Train the model on the prepared dataset
model.train(
    data=dataset_yaml_path,
    epochs=1,      # Set to a smaller number for testing, increase for better training
    imgsz=416,
    batch=8
)

# 4. Validate YOLO Model
model.val(data=dataset_yaml_path, imgsz=416, epochs=3)

# Load the generate best model
model = YOLO(basePath + "runs/detect/train/weights/best.pt")

# Export the model to TFLite format
model.export(format="tflite")  # creates 'yolo11n_float32.tflite'

print("Generate model file is " + basePath + "runs/detect/train/weights/best_saved_model/best_float16.tflite")
