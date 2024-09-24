import os
import json
import random
import shutil

# Path to your images and annotation file
image_dir = 'Dataset\Pedestrian_dataset_for_internship_assignment'
annotation_file = r'Dataset\random_sample_mavi_2_gt.json'

# Load COCO annotations (JSON file)
with open(annotation_file, 'r') as f:
    coco_data = json.load(f)

# Get image, annotation, and category data
images = coco_data['images']
annotations = coco_data['annotations']
categories = coco_data['categories']

# Split ratio for train and validation sets
split_ratio = 0.8

# Shuffle the images randomly
random.shuffle(images)

# Split images into train and validation sets
split_index = int(len(images) * split_ratio)
train_images = images[:split_index]
val_images = images[split_index:]

# Create a set of image ids for fast lookup
train_image_ids = {img['id'] for img in train_images}
val_image_ids = {img['id'] for img in val_images}

# Split annotations based on the image ids in each set
train_annotations = [ann for ann in annotations if ann['image_id'] in train_image_ids]
val_annotations = [ann for ann in annotations if ann['image_id'] in val_image_ids]

# Create the train and validation JSON structures
train_data = {
    'images': train_images,
    'annotations': train_annotations,
    'categories': categories
}

val_data = {
    'images': val_images,
    'annotations': val_annotations,
    'categories': categories
}

# Save the train and validation annotations as separate JSON files
train_annotation_file = r'Dataset\\train_annotations.json'
val_annotation_file = r'Dataset\\val_annotations.json'

with open(train_annotation_file, 'w') as f:
    json.dump(train_data, f)

with open(val_annotation_file, 'w') as f:
    json.dump(val_data, f)

print(f"Train and validation annotations have been saved to {train_annotation_file} and {val_annotation_file}")

# Optional: Move or copy images to separate train and val folders
train_image_dir = 'Dataset\\train_images'
val_image_dir = 'Dataset\\val_images'

# Create directories if they don't exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)

# Move or copy images to the corresponding folders
for image in train_images:
    src_path = os.path.join(image_dir, image['file_name'])
    dest_path = os.path.join(train_image_dir, image['file_name'])
    shutil.copy(src_path, dest_path)  # Use shutil.move() if you want to move instead of copying

for image in val_images:
    src_path = os.path.join(image_dir, image['file_name'])
    dest_path = os.path.join(val_image_dir, image['file_name'])
    shutil.copy(src_path, dest_path)

print(f"Images have been copied to {train_image_dir} and {val_image_dir}")
