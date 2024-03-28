import numpy as np
import os
from PIL import Image
import numpy as np
import os
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import (
    InceptionV3, VGG16, EfficientNetB0, MobileNetV2, ResNet50,
    Xception, DenseNet121
)
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inceptionv3
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenetv2
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.xception import preprocess_input as preprocess_xception
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tensorflow.keras.applications import NASNetLarge, InceptionResNetV2
from tensorflow.keras.applications.nasnet import preprocess_input as preprocess_nasnet
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inceptionresnetv2

def preprocess_image(file_path, model_preprocess, target_size):
    img = Image.open(file_path)
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = image.img_to_array(img)
    img_array = model_preprocess(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def extract_features(model, preprocess_function, target_size, folder_path, limit=1000):
    features_list = []
    filenames = os.listdir(folder_path)[:limit]
    for filename in filenames:
        file_path = os.path.join(folder_path, filename)
        if file_path.lower().endswith('.tif'):
            img_array = preprocess_image(file_path, preprocess_function, target_size)
            features = model.predict(img_array)
            features_list.append(features.flatten())
    return np.array(features_list)


# Further extended models_to_test list
models_to_test = [
    (VGG16, preprocess_vgg16, (224, 224)),
    (InceptionResNetV2, preprocess_inceptionresnetv2, (299, 299)),
    (DenseNet121, preprocess_densenet, (224, 224)),
    (Xception, preprocess_xception, (299, 299))
]


folder_path = r'C:\Users\bollo\Desktop\machine learning\cleaned'

for model_func, preprocess_func, size in models_to_test:
    print(f"Testing {model_func.__name__}...")
    base_model = model_func(include_top=False, weights='imagenet', pooling='avg')
    features = extract_features(base_model, preprocess_func, size, folder_path)

    # Clustering
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=22)
    kmeans.fit(features)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(features, kmeans.labels_)
    print(f"Model: {model_func.__name__}, Silhouette Score: {silhouette_avg}\n")
