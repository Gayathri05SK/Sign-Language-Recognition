import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Define the path to your dataset folder
data_dir = 'Data'

# Initialize variables for analysis
class_counts = Counter()
image_sizes = []
intensity_distributions = []
missing_images = []

# Loop through each class folder (letters and digits)
for class_folder in os.listdir(data_dir):
    class_folder_path = os.path.join(data_dir, class_folder)
    
    if os.path.isdir(class_folder_path):
        class_counts[class_folder] += len(os.listdir(class_folder_path))  # Class distribution count
        
        for image_file in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, image_file)
            
            # Try to open and process the image
            try:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                # Check if the image is valid
                if img is None:
                    missing_images.append(image_path)
                    continue
                
                # Image size
                image_sizes.append(img.shape)
                
                # Image intensity distribution
                intensity_distributions.append(img.flatten())
                
            except Exception as e:
                missing_images.append(image_path)

# Class Distribution Plot
plt.figure(figsize=(10, 5))
plt.bar(class_counts.keys(), class_counts.values())
plt.title('Class Distribution')
plt.xlabel('Classes')
plt.ylabel('Number of Images')
plt.xticks(rotation=90)
plt.show()

# Image Size and Resolution Analysis
unique_sizes = Counter(image_sizes)
print("Unique image sizes and counts:", unique_sizes)

# Image Intensity Distribution Plot
intensity_distributions = np.concatenate(intensity_distributions, axis=0)
plt.figure(figsize=(10, 5))
plt.hist(intensity_distributions, bins=50, color='blue', alpha=0.7)
plt.title('Image Intensity Distribution')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

# Missing/Corrupt Data Check
if missing_images:
    print(f"Missing or corrupt images: {len(missing_images)}")
    print("Example paths of missing/corrupt images:", missing_images[:5])
else:
    print("No missing or corrupt images found.")
