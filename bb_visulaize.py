import os
import json
import cv2
import matplotlib.pyplot as plt

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

# Create a dictionary to map image ids to image filenames
image_dict = {image['id']: image['file_name'] for image in images}

# Create a dictionary to map category ids to category names
category_dict = {category['id']: category['name'] for category in categories}

# Number of images you want to display per figure
num_images_per_row = 4
num_images_per_col = 2
total_images = num_images_per_row * num_images_per_col

# Iterate over images in batches
for i in range(0, len(images), total_images):
    fig, axes = plt.subplots(num_images_per_col, num_images_per_row, figsize=(20, 10))
    axes = axes.flatten()  # To access subplots easily as a 1D array

    # Plot images with bounding boxes and category names
    for j, ax in enumerate(axes):
        if i + j >= len(images):  # Avoid index error in the last batch
            break
        
        # Get the image and its annotations
        image_info = images[i + j]
        image_id = image_info['id']
        img_path = os.path.join(image_dir, image_dict[image_id])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get annotations for this image
        img_annotations = [ann for ann in annotations if ann['image_id'] == image_id]

        # Draw bounding boxes and category names on the image
        for ann in img_annotations:
            bbox = ann['bbox']
            category_id = ann['category_id']
            category_name = category_dict[category_id]

            x, y, width, height = bbox
            start_point = (int(x), int(y))
            end_point = (int(x + width), int(y + height))
            color = (255, 0, 0)  # Red for bounding box
            thickness = 2
            img = cv2.rectangle(img, start_point, end_point, color, thickness)

            # Put the category name above the bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            text_size = cv2.getTextSize(category_name, font, font_scale, font_thickness)[0]
            text_start = (int(x), int(y) - 5)  # Position text slightly above the bounding box
            img = cv2.putText(img, category_name, text_start, font, font_scale, (255, 233, 0), font_thickness)

        # Display image in the current subplot
        ax.imshow(img)
        ax.axis('off')  # Hide the axes

    plt.tight_layout()
    plt.show()
