import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def preprocess_image(image_path, size=(224, 224)):
    """Load an image, resize it, convert it to grayscale, and flatten it."""
    with Image.open(image_path) as img:
        img = img.resize(size).convert('L')  # Convert to grayscale
        img_array = np.array(img).flatten()  # Flatten the image
    return img_array

def load_labeled_data(csv_path):
    """Load labeled data from a CSV file and preprocess images."""
    df = pd.read_csv(csv_path)
    X_labeled = np.array([preprocess_image(path) for path in df['ImagePath']])
    y_labeled = convert_labels_to_int(df['Label'].values)  # Conversion inside the function
    return X_labeled, y_labeled

def load_unlabeled_data(unlabeled_folder):
    """Load and preprocess all images in the unlabeled folder."""
    image_paths = [os.path.join(unlabeled_folder, f) for f in os.listdir(unlabeled_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
    X_unlabeled = np.array([preprocess_image(path) for path in image_paths])
    return X_unlabeled, image_paths

def convert_labels_to_int(labels):
    """Converts string labels to integers if they are not already integers."""
    if isinstance(labels[0], str):
        unique_labels = sorted(set(labels))
        label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        return np.array([label_to_int[label] for label in labels])
    else:
        return labels

def select_high_confidence_predictions(model, X_unlabeled, threshold=0.75):
    """Select high-confidence predictions."""
    probabilities = model.predict_proba(X_unlabeled)
    max_probs = np.max(probabilities, axis=1)
    high_confidence_indices = np.where(max_probs > threshold)[0]
    high_confidence_labels = np.argmax(probabilities, axis=1)[high_confidence_indices]
    return high_confidence_indices, high_confidence_labels


# Load and preprocess the labeled data
csv_path = r'C:\Users\bollo\Desktop\machine learning\labeled_images.csv'
X_labeled, y_labeled = load_labeled_data(csv_path)

# Split the labeled data into train, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Validate the initial model
val_predictions = model.predict(X_val)
print(f"Initial Validation Accuracy: {accuracy_score(y_val, val_predictions)}")

# Load and preprocess unlabeled data (unlabeled_folder path and load_unlabeled_data function remain unchanged)

unlabeled_folder = r'C:\Users\bollo\Desktop\machine learning\cleaned'
X_unlabeled, _ = load_unlabeled_data(unlabeled_folder)

# Function for selecting high-confidence predictions (select_high_confidence_predictions) remains unchanged

# Predict on unlabeled data and select high-confidence predictions
high_conf_indices, high_conf_labels = select_high_confidence_predictions(model, X_unlabeled)

# Prepare the expanded training dataset
X_train_expanded = np.concatenate([X_train, X_unlabeled[high_conf_indices]])
y_train_expanded = np.concatenate([y_train, high_conf_labels])

# Retrain the model with the expanded dataset
model.fit(X_train_expanded, y_train_expanded)

# Final evaluation on the test set
test_predictions = model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, test_predictions)}")