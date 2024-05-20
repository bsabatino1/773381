# Résumé Classification and OCR Project

### Team Members
- Edoardo Bollati 773381
  
- Fabrizio Borrelli 789121

- Benedetta Sabatino 781701

## [Section 1] Introduction
The aim of this project is to construct a model capable of accurately identifying the category to which each résumé belongs. This involves a thorough analysis of the dataset ensuring its cleanliness and developing a robust classification model. Additionally, an optional task is to employ OCR models to extract pertinent information from the documents. The dataset comprises 5000 scanned résumé images in TIF format.

## [Section 2] Methods

### Environment setup
To recreate this project in environments other than Google Colab, refer to the `requirements.txt` file available in the GitHub repository. Alternatively, all necessary installations can be found within the notebook itself.

### Pre-Processing

- #### Image Denoising
  This script processes the images to enhance text readability and improve the efficiency and accuracy of subsequent text analysis tasks. The primary goal is to denoise the images by retaining only the textual content.
  
  **Input**: the original dataset.
  
   **Output**: Cleaned images saved in a specified output folder with non-text areas turned white, making the text more prominent.
  
  **Features**:
  - Extracts text and bounding boxes using Optical Character Recognition (OCR).
  - Removes non-textual elements from images, turning these areas white.
  - Processes multiple images in batch mode.

- #### Creation of black boxes
  This script uses OCR and clustering algorithms to identify, merge, and highlight text regions. The processed images, where text areas are blacked out, are then saved in an output folder.
  
  **Input**: output folder of image denoising
  **Output**: Processed images saved with text areas blacked out to highlight text regions.
  
  **Features**:
  - Extracts text and bounding boxes using OCR.
  - Removes non-textual elements by blacking out text areas.
  - Groups and merges nearby text boxes using clustering algorithms.
  - Processes multiple images in batch mode.

### Model Tuning

- #### Model search
  This script processes images to extract features, reduce dimensionality, cluster, and visualize results using pre-trained CNN models and various clustering methods. The resulting best combination is EfficientNetB0, UMAP, and KMeans.
  
   **Input**: the output folder resulting from the creation of black boxes.
  
  **Output**: Processed images saved in clearly labeled folders organized by the combination of model, dimensionality reduction, and clustering method used.
  
  **Features**:
  - Uses pre-trained CNN models (VGG16, InceptionV3, InceptionResNetV2, EfficientNetB0) for robust feature extraction.
  - Dimensionality Reduction: Applies PCA, t-SNE, and UMAP to explore the data's structure.
  - Clustering: Uses KMeans and Agglomerative Clustering to identify meaningful groups.
  - Output Structure: Saves images in labeled folders for easy analysis and presentation.

- #### Grid search
  This script combines deep learning features from EfficientNet with traditional image processing techniques (LBP and HOG) to capture comprehensive image information. UMAP is used for dimensionality reduction and KMeans for clustering. The script ensures optimal clustering performance through hyperparameter tuning.
  
  **Input**: the output folder resulting from the creation of black boxes.
  
  **Output**: Processed images saved in clearly labeled folders organized by the combination of UMAP and KMeans parameters used and the best combination.
  
  **Features**:
  - Feature Extraction: Uses EfficientNet for high-quality features and LBP/HOG for additional texture and shape information.
  - Data Augmentation: Applies various transformations such as rotation (up to 20 degrees), width and height shifts (up to 20%), shear transformations (up to 20%), zoom (up to 20%), and horizontal flips to enhance the diversity of the training data, ensuring robustness and improving model generalization.
  - Dimensionality Reduction: Applies UMAP to reduce dimensionality while preserving data structure.
  - Clustering: Uses KMeans to group images into meaningful clusters.
  - Hyperparameter Optimization: Utilizes GridSearchCV for optimal parameter selection.
  - Output Structure: Saves images in labeled folders for easy analysis and presentation.

### Models

- #### First approach: Unsupervised Model
  This script provides a robust approach to image feature extraction, data augmentation, dimensionality reduction, clustering, and visualization by combining traditional image processing techniques with deep learning features. EfficientNet is used for high-quality feature extraction while custom features (LBP and HOG) capture additional texture and shape information. UMAP is applied for dimensionality reduction and KMeans for clustering.
  
  **Input**: A folder containing black boxed images
  
  **Output**: Processed images saved in clearly labeled folders organized by the combination of UMAP and KMeans parameters used.
  
  **Features**:
  - Feature Extraction: Uses EfficientNet for high-quality features and custom features (LBP and HOG) for texture and edge information.
  - Data Augmentation: Applies various transformations such as rotation (up to 20 degrees), width and height shifts (up to 20%), shear transformations (up to 20%), zoom (up to 20%), and horizontal flips to enhance the diversity of the training data.
  - Dimensionality Reduction: Applies UMAP to reduce dimensionality while preserving data structure.
  - Clustering: Uses KMeans to group images into meaningful clusters.
  - Hyperparameter Optimization: Utilizes grid search to find the best parameters for UMAP and KMeans.
  - Output Structure: Saves images in labeled folders for easy analysis and presentation.

- #### Second approach: Semi-supervised model
  This approach involves two scripts: one for manual labeling of images and another for semi-supervised clustering using EfficientNetB0, LBP, HOG, UMAP, and COP-KMeans.
  
  **Features**:
    - Interactive UI for labeling images.
    - Saves labels to a CSV file for further analysis.
    - Uses EfficientNetB0 for deep feature extraction and custom features (LBP and HOG) for additional information.
    - Applies UMAP for dimensionality reduction and COP-KMeans for constrained clustering.
    - Utilizes grid search for hyperparameter optimization.
    - Saves clustering results into folders and prints evaluation metrics.

- #### Third Approach: Unsupervised Model with BERT and LayoutLM
  This approach focuses on unsupervised clustering of document images. The process involves feature extraction using BERT and LayoutLM models, dimensionality reduction using PCA, and clustering with KMeans.
  
  **Key Steps and Features**:
  1. Text and Layout Feature Extraction:
     - Text Features with BERT: Utilizes a pre-trained BERT model to extract semantic features from the text within the document images.
     - Layout Features with LayoutLM: Employs a pre-trained LayoutLM model to capture both textual and spatial information.
  2. Dimensionality Reduction:
     - Principal Component Analysis (PCA): Reduces the dimensionality of the combined text and layout feature vectors.
  3. Clustering:
     - KMeans Clustering: Groups the document images into clusters based on the reduced feature vectors. The optimal number of clusters is determined using the silhouette score.
  
  **Features**:
  - BERT for Text Feature Extraction.
  - LayoutLM for Layout Feature Extraction.
  - PCA for Dimensionality Reduction.
  - KMeans Clustering with Silhouette Analysis.

### OCR

- #### DnCNN
  Image denoising is essential for enhancing the quality of images in various applications such as document digitization, medical imaging, and photography. The Deep Convolutional Neural Network (DnCNN) effectively removes noise by leveraging deep learning techniques.
  
  - **Inputs**: preprocessed images from image denoising
  - **Outputs**: denoised and improved images
  
  **Features**:
  - Dataset Augmentation: Added 300 scanned document images to the dataset.
  - Model Architecture: Increased the number of layers to 20.
  - Early Stopping: Training stops when PSNR no longer improves.
  - Image Transformation Pipeline: Converts images to grayscale, transforms to tensors, and normalizes.
  
- #### OCR


## [Section 3] Experimental Design

### Unsupervised Model
**Advantages**:
The unsupervised model is advantageous due to its ability to extract and utilize a rich set of features without the need for labeled data. By employing EfficientNet for high-quality feature extraction and incorporating traditional techniques like Local Binary Patterns (LBP) and Histogram of Oriented Gradients (HOG), this approach captures a comprehensive set of image characteristics, including deep learning features, textures, and shapes. The use of UMAP for dimensionality reduction allows for maintaining the intrinsic structure of the data while simplifying the feature space, facilitating effective visualization and clustering with KMeans. Additionally, the hyperparameter optimization through grid search ensures that the best model parameters are selected, enhancing the performance and accuracy of the clustering process.

**Disadvantages**:
However, this approach has certain limitations. Being unsupervised, it might not always produce meaningful clusters if the inherent patterns in the data are not strong enough, leading to potentially less interpretable results. The reliance on high computational resources for running EfficientNet and performing grid search for optimal parameters can be intensive, making it less feasible for very large datasets or environments with limited computational power. Furthermore, without labeled data, there is no straightforward way to validate the accuracy and relevance of the discovered clusters, potentially limiting the usefulness of the results for specific applications.

### Semi-Supervised Model
**Advantages**:
The semi-supervised model combines the strengths of both supervised and unsupervised learning techniques, offering a balanced approach to image clustering. The manual labeling script provides an interactive user interface for labeling images, which helps create a high-quality labeled dataset. This labeled data can guide the clustering process, incorporating domain knowledge to improve accuracy and relevance. EfficientNetB0 ensures robust feature extraction, while UMAP and COP-KMeans (Constrained KMeans) allow for effective dimensionality reduction and constrained clustering. The inclusion of constraints (must-link and cannot-link) derived from labeled data leads to more meaningful and accurate clusters, and the use of grid search for hyperparameter optimization further refines the model’s performance.

**Disadvantages**:
Despite its advantages, the semi-supervised model also has drawbacks. The process of manual labeling can be time-consuming and labor-intensive, especially for large datasets, which can slow down the workflow. This approach still requires significant computational resources for deep feature extraction and hyperparameter tuning. Additionally, while the use of constraints can improve clustering accuracy, it also adds complexity to the model and can introduce bias if the initial labeled data is not representative of the entire dataset. This could potentially limit the generalizability of the clustering results. Human biases introduced during manual labeling may not align with the feature extraction process of the model, potentially leading to less effective clustering. Furthermore, the lack of a strong layout or structure for integrating labeled and unlabeled data can make the semi-supervised approach less efficient and more challenging to implement.

A notable issue observed is that some images appear in more than one cluster. This could be attributed to the use of COP-KMeans (Constrained KMeans), as the constraints may lead to overlapping clusters or force certain images to satisfy multiple constraints, resulting in them being included in multiple clusters.

### Unsupervised Model with BERT and LayoutLM
**Advantages**:
The unsupervised model using BERT and LayoutLM offers several advantages for clustering document images without labeled data. It eliminates the need for labeled training data, allowing the model to identify patterns autonomously. The combination of BERT’s contextual text understanding and LayoutLM’s layout information results in a comprehensive document representation, enhancing clustering quality.
Principal Component Analysis (PCA) optimizes the process by retaining essential features and reducing computational load, making clustering more efficient for large datasets. The KMeans algorithm, guided by silhouette scores, ensures well-defined clusters. This method's versatility makes it applicable to various document types, providing a robust solution for document clustering.

**Disadvantages**:
Despite its strengths, the model has several challenges. High computational complexity for feature extraction and clustering requires substantial resources, which can be a bottleneck for large datasets. Even with efficient memory management, handling very large datasets can be problematic, slowing down the workflow.
The absence of labeled data can lead to less accurate clustering, as there are no ground truth references to guide the model.


## Evaluation Metrics

1. ### Silhouette Score
  The silhouette score ranges from -1 to 1 and evaluates how similar an object is to its own cluster compared to other clusters. Higher values indicate that objects are well-clustered and distinct from other clusters, while lower or negative values suggest that objects may be improperly clustered or overlap significantly with other clusters.
  
  Usefulness:
  -	Cohesion and Separation: It provides an overall measure of how tightly grouped the data points are within their clusters and how distinct each cluster is from others.
  -	Quality Assessment: The score helps in determining the effectiveness of the clustering algorithm in creating well-defined clusters.


2. ### Davies-Bouldin Index
  The Davies-Bouldin index measures the average similarity ratio of each cluster with its most similar cluster, with lower values indicating better clustering. It assesses how compact each cluster is and how far apart different clusters are from each other.
  
  Usefulness:
  -	Cluster Compactness and Separation: This index helps in understanding the degree of overlap between clusters and how compact the clusters are.
  -	Comparative Analysis: It allows for the comparison of clustering algorithms by indicating which one produces more distinct and compact clusters.


3. ### Calinski-Harabasz Index
  The Calinski-Harabasz index, also known as the Variance Ratio Criterion, measures the ratio of between-cluster dispersion to within-cluster dispersion. Higher values suggest better clustering performance, indicating that the clusters are well-separated and have low within-cluster variance.
  
  Usefulness:
  -	Dispersion Analysis: It evaluates the extent to which clusters are distinct from each other and how concentrated the points are within each cluster.
  -	Clustering Effectiveness: This index is useful for comparing the performance of different clustering models and determining which one provides the most distinct and cohesive clusters.


## Results

### Unsupervised model results
1. **Average Silhouette Score**: 0.64
   
   A silhouette score of 0.64 indicates a good clustering structure. This means that, on average, the samples are well-matched to their own clusters and clearly distinct from neighboring clusters.

3. **Davies-Bouldin Index**: 0.58
   
   A value of 0.58 indicates relatively low intra-cluster variance compared to inter-cluster variance. This means the clusters are compact and the separation between clusters is significant.

5. **Calinski-Harabasz Index**: 35593.91
   
   A high score of 35,593.91 suggests that the clusters are both compact and well-separated. This indicates that data points within each cluster are close to each other, while the clusters themselves are distinctly different from one another.


The silhouette score of 0.64, along with the low Davies-Bouldin Index of 0.58, indicates strong intra-cluster cohesion and distinct inter-cluster separation. The exceptionally high Calinski-Harabasz Index of 35,593.91 further confirms the effectiveness of the clustering solution, indicating that the clusters are well-defined and well-separated. Together, these results suggest that the unsupervised model using EfficientNetB0, KMeans, and UMAP is highly effective in creating meaningful and distinct clusters from the image data. 

## AGGIUNGERE CONSIDERAZIONE SUL CLUSTER


<img width="297" alt="image" src="https://github.com/bsabatino1/ML_PROJECT_773381_group2/assets/94707288/08ed3909-1f9f-472c-9367-e92b0eaaa5f3">

### Semi-supervised model results
1. **Silhouette Score**: 0.64
   
   A silhouette score of 0.64 indicates good intra-cluster cohesion and inter-cluster separation. This means that, on average, the samples are well-matched to their own clusters and clearly distinct from neighboring clusters.

2. **Davies-Bouldin Index**: 5.01
   
   A value of 5.01 suggests that some clusters might be too close to each other or not compact enough. This higher value may be due to human bias in labeled data, which could affect the clustering performance.

3. **Calinski-Harabasz Index**: 6834.43

   A score of 6,834.43 reflects a high level of cluster definition and separation. This indicates that data points within each cluster are relatively close to each other, while the clusters themselves are distinctly different from one another.


The semi-supervised results present a mixed outcome. The silhouette score of 0.64 indicates that the overall clustering structure is good, with strong intra-cluster cohesion and clear inter-cluster separation. However, the Davies-Bouldin Index of 5.01 is relatively high, suggesting potential issues with cluster compactness and separation. This could be attributed to human biases introduced during the manual labeling process, which may not align well with the model's feature extraction. The Calinski-Harabasz Index of 6,834.43, while lower than the unsupervised result, still indicates well-defined and separated clusters.

The combination of these indices suggests that while the semi-supervised approach benefits from incorporating labeled data to guide the clustering process, it also introduces complexities and potential biases that can affect the clustering quality. The constraints used in the semi-supervised model might lead to some images being included in multiple clusters, further complicating the clustering results. Overall, the semi-supervised model shows promise but requires careful handling of labeled data and constraints to ensure robust and accurate clustering outcomes.


<img width="385" alt="image" src="https://github.com/bsabatino1/ML_PROJECT_773381_group2/assets/94707288/a503a8f7-9908-49aa-a69b-7ccb9890b58f">

### MODELLO FABBRI

### OCR RESULTS

## Conclusions

### Takeaway Points

### Future Work
- Higher computational power
- Implementation of BERT on OCR
- Addressing unanswered questions and next steps
