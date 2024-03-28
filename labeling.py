import cv2
import os
import pandas as pd
import random

# Define your image directory and output CSV file name
image_dir = r'C:\Users\bollo\Desktop\machine learning\cleaned'
output_csv = 'labeled_images.csv'

# Get a list of image file paths
all_images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')]

# Select a random 25% subset of the images for labeling
subset_size = len(all_images) // 100  # 25%
images = random.sample(all_images, subset_size)

# Container for labels
labels = []

print("Labeling images. Close the image window and enter the label in the console when prompted.")

# Display each image in the subset and prompt for input
for image_path in images:
    # Read and display the image
    img = cv2.imread(image_path)
    cv2.imshow('Image', img)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed to close the image window

    # Prompt the user for a label
    label = input(f"Label for {image_path} ('cv' or 'not cv'): ")
    labels.append(label)

    # Close the image window
    cv2.destroyAllWindows()

# Save the labels to a CSV file
df = pd.DataFrame(list(zip(images, labels)), columns=['ImagePath', 'Label'])
df.to_csv(output_csv, index=False)

print("Labeling complete. Labels saved to:", output_csv)
