import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def advanced_text_readability(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply adaptive thresholding
    img_adaptive_threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 15, 5)

    # Apply morphological operations to clean up the background
    kernel = np.ones((1, 1), np.uint8)
    img_morph = cv2.morphologyEx(img_adaptive_threshold, cv2.MORPH_OPEN, kernel)
    img_morph = cv2.dilate(img_morph, kernel, iterations=1)

    # Displaying the processed image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_morph, cmap='gray')
    plt.title('Enhanced for Readability')
    plt.axis('off')

    plt.show()

# Example usage (replace 'path_to_your_image.tif' with the actual path to your image)
advanced_text_readability(r'C:\Users\bollo\Desktop\machine learning\resume\0000000869.tif')
