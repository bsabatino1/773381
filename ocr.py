"""#sistema e fai runnare questo codice
import os
import pytesseract

# Function to perform OCR on images in a folder and save the text to another folder
def perform_ocr_and_save(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(".tif") or filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            text = pytesseract.image_to_string(image_path, lang='eng', config='--oem 3 --psm 3')

            # Create a text file with the same filename in the output folder
            output_filename = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(output_folder, output_filename)
            with open(output_path, "w") as text_file:
                text_file.write(text)

# Example folder paths - replace these with the paths to your input and output folders
input_folder = "/path/to/input/folder"
output_folder = "/path/to/output/folder"

# Perform OCR on images in the input folder and save the text to the output folder
perform_ocr_and_save(input_folder,output_folder)"""
import pytesseract
from PIL import Image, ImageOps, ImageFilter
import glob

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_text_from_image(image_path, custom_config=r'--oem 3 --psm 6 -l eng'):
    """
    Extracts text from a given image using pytesseract with custom configuration.
    Includes preprocessing steps to potentially improve OCR accuracy.

    :param image_path: Path to the image file.
    :param custom_config: Custom configuration string for pytesseract.
    :return: Extracted text as a string.
    """
    # Load the image from the specified path
    image = Image.open(image_path)

    # Apply preprocessing
    # Apply a median filter to reduce noise
    filtered_image = image.filter(ImageFilter.MedianFilter(size=1))
    # Enhance contrast
    enhanced_image = ImageOps.autocontrast(filtered_image, cutoff=3)

    # Extract text using pytesseract with custom configurations
    text = pytesseract.image_to_string(enhanced_image, config=custom_config)

    return text


def extract_texts_from_folder(folder_path, limit=None):
    """
    Extracts texts from all or a limited number of images in a specified folder
    using optimized OCR parameters and preprocessing.

    :param folder_path: Path to the folder containing image files.
    :param limit: Optional. The maximum number of images to process. If None, all images are processed.
    :return: A dictionary with image file paths as keys and extracted texts as values.
    """
    # Find all image files in the folder
    image_paths = glob.glob(f"{folder_path}/*.tif")  # Adjust the pattern as needed

    # If a limit is specified, only process up to that number of images
    if limit is not None:
        image_paths = image_paths[:limit]

    # Extract text from each image using the custom OCR function
    texts = {}
    for image_path in image_paths:
        text = extract_text_from_image(image_path)
        texts[image_path] = text

    return texts


folder_path = r'C:\Users\bollo\Desktop\machine learning\cleaned'

#limit to the number of images to process for testing
limit = 1

# Extract text from images in the folder
extracted_texts = extract_texts_from_folder(folder_path, limit=limit)

# Print the extracted text for each image
for path, text in extracted_texts.items():
    print(f"Text from {path}:")
    print(text)
    print("------")
