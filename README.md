# Résumé Classification and OCR Project

### Team Members
- Edoardo Bollati 773381
  
- Fabrizio Borrelli 789121

- Benedetta Sabatino 781701

## [Section 1] Introduction
The aim of this project is to construct a model capable of accurately identifying the category to which each résumé belongs. This involves a comprehensive analysis of the dataset ensuring its cleanliness and developing a robust classification model. Additionally, an optional task is to employ OCR models to extract pertinent information from the documents. The dataset comprises 5000 scanned résumé images in TIF format.

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
  This script uses Optical Character Recognition (OCR) and clustering algorithms, it identifies, merges, and highlights text regions. The processed images, where text areas are blacked out, are then saved in an output folder.
  
  **Input**: output folder of image denoising
  
  **Output**: Processed images saved with text areas blacked out to highlight text regions.
  
  **Features**:
  - Extracts text and bounding boxes using OCR.
  - Removes non-textual elements by blacking out text areas.
  - Groups and merges nearby text boxes using clustering algorithms.
  - Processes multiple images in batch mode.

### Model Tuning

- #### Model search
  This script processes images to extract features, reduce dimensionality, cluster, and visualize results using pre-trained CNN models and various clustering methods. By leveraging state-of-the-art pre-trained CNN models, it ensures robust feature extraction. The combination of multiple dimensionality reduction and clustering techniques allows for a thorough exploration of the image dataset, identifying the most meaningful groupings and selecting the best combination for further analysis. The resulting best combination is EfficientNetB0, UMAP, and KMeans.

   **Input**: the output folder resulting from the creation of black boxes.
  
  **Output**: Processed images saved in clearly labeled folders organized by the combination of model, dimensionality reduction, and clustering method used.
  
  **Features**:
  - Uses pre-trained CNN models (VGG16, InceptionV3, InceptionResNetV2, EfficientNetB0) for robust feature extraction.
  - Dimensionality Reduction: Applies PCA, t-SNE, and UMAP to explore the data's structure.
  - Clustering: Uses KMeans and Agglomerative Clustering to identify meaningful groups.
  - Output Structure: Saves images in labeled folders for easy analysis and presentation.

- #### Grid search
  This script combines deep learning features from EfficientNet with traditional image processing techniques (LBP and HOG) to capture comprehensive image information. UMAP is used for dimensionality reduction and KMeans for clustering. The script ensures optimal clustering performance through hyperparameter tuning and organizes the output for easy analysis and presentation, providing the best combination which is:
  
  -	UMAP Parameters: n_neighbors=50, min_dist=0.1, metric='euclidean', spread=1.0, n_epochs=200, negative_sample_rate=0.1.
  - KMeans Parameters: n_clusters=4, init='k-means++', n_init=10, tol=0.0001, algorithm='lloyd'.


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
  This script provides a robust approach to image feature extraction, data augmentation, dimensionality reduction, clustering, and visualization by combining traditional image processing techniques with deep learning features. EfficientNet is used for high-quality feature extraction, while custom features (LBP and HOG) capture additional texture and shape information. UMAP is applied for dimensionality reduction and KMeans for clustering, allowing thorough exploration of the image dataset to identify meaningful groupings. The chosen parameters result from extensive model and parameter search, ensuring the best performance.

  **Input**: A folder containing black boxed images
  
  **Output**: Processed images saved in clearly labeled folders organized by the combination of UMAP and KMeans parameters used.
  
  **Features**:
  - Feature Extraction: Uses EfficientNet for high-quality features and custom features (LBP and HOG) for texture and edge information.
  - Data Augmentation: Applies various transformations such as rotation (up to 20 degrees), width and height shifts (up to 20%), shear transformations (up to 20%), zoom (up to 20%), and horizontal flips to enhance the diversity of the training data. This augmentation step ensures robustness and improves model generalization by simulating a variety of real-world conditions.
  - Dimensionality Reduction: Applies UMAP to reduce dimensionality while preserving data structure.
  - Clustering: Uses KMeans to group images into meaningful clusters.
  - Hyperparameter Optimization: Utilizes grid search to find the best parameters for UMAP and KMeans.
  - Output Structure: Saves images in labeled folders for easy analysis and presentation.

- #### Second approach: Semi-supervised model
  This approach involves two scripts, the first one is designed to facilitate manual labeling of images. It provides an interactive user interface using OpenCV for image display and `ipywidgets` for creating dropdown menus and buttons. The process includes loading and displaying images, allowing the user to select labels from a dropdown menu, and saving the progress periodically. The second script employs semi-supervised clustering to classify images using a combination of EfficientNetB0 for deep feature extraction and custom features (LBP and HOG). It incorporates data augmentation, dimensionality reduction with UMAP, and clustering via the COP-KMeans algorithm, which utilizes must-link and cannot-link constraints derived from labeled data. The clustering results are saved into organized folders and a CSV file. Key evaluation metrics are calculated to gauge the quality of the clustering.

  **Features**:
    - Interactive UI for labeling images.
    - Saves labels to a CSV file for further analysis.
    - Uses EfficientNetB0 for deep feature extraction and custom features (LBP and HOG) for additional information.
    - Applies UMAP for dimensionality reduction and COP-KMeans for constrained clustering.
    - Utilizes grid search for hyperparameter optimization.
    - Saves clustering results into folders and prints evaluation metrics.

- #### Third Approach: Unsupervised Model with BERT and LayoutLM
  This approach focuses on unsupervised clustering of document images. The process is divided into several key steps, involving feature extraction using BERT and LayoutLM models, dimensionality reduction using PCA, and clustering with KMeans. This method aims to cluster document images relying entirely on the intrinsic properties of the images and extracted features.


  **Key Steps and Features**:
  1. **Text and Layout Feature Extraction**:
     - Text Features with BERT: Utilizes a pre-trained BERT model to extract semantic features from the text within the document images. BERT's attention mechanism captures rich contextual information.
     - Layout Features with LayoutLM: Employs a pre-trained LayoutLM model to capture both textual and spatial information. LayoutLM extends the BERT architecture to include layout information, which is crucial for understanding the structure of documents.
  2. **Dimensionality Reduction**:
     - Principal Component Analysis (PCA): Reduces the dimensionality of the combined text and layout feature vectors. PCA helps in retaining the most important features while reducing the computational complexity of the clustering process.
  3. **Clustering**:
     - KMeans Clustering: Groups the document images into clusters based on the reduced feature vectors. The optimal number of clusters is determined using the silhouette score, which measures the quality of the clustering.
  
  **Features**:
    - BERT for Text Feature Extraction: Leverages the power of BERT to extract high-quality semantic features from the text within documents.
    - LayoutLM for Layout Feature Extraction: Captures both textual and spatial information, providing a comprehensive feature representation of the documents.
    - PCA for Dimensionality Reduction: Reduces the feature dimensionality, making the clustering process more efficient and manageable.
    - KMeans Clustering with Silhouette Analysis: Determines the optimal number of clusters using silhouette scores and performs clustering accordingly.

### OCR

- #### DnCNN
  
  Image denoising is essential for enhancing the quality of images, the Deep Convolutional Neural Network (DnCNN) effectively removes noise by leveraging deep learning techniques, including residual learning and batch normalization, for efficient and stable performance.
  
  - **Inputs**: preprocessed images from image denoising
    
  - **Outputs**: denoised and improved images
    
  
  **Features**:
  - **CNN Training Dataset Augmentation**: Added 300 scanned document images to the dataset. This improves model performance by teaching it to handle real-world noise patterns specific to scanned documents.
  - **Model Architecture**: Increased the number of layers to 20. Captures more complex noise patterns for better denoising performance.
  - **Early Stopping**: Training stops when PSNR no longer improves. This prevents overfitting and saves computational resources by stopping at optimal performance.
  - **Image Transformation Pipeline**: Converts images to grayscale, transforms to tensors, and normalizes. It prepares images effectively for the DnCNN model.

  
- #### OCR
    In this section a systematic approach of preprocessing, text extraction, spelling correction, and validation against English words ensures that the extracted text from images is as accurate and meaningful as possible. By comparing different preprocessing techniques and utilizing advanced NLP tools, the project aims to optimize OCR performance, providing valuable insights into the effectiveness of various methods in enhancing text extraction accuracy.
  
  - **Inputs**: images processed by the DnCNN.

  - **Outputs**: json file with the text extracted.

  
  **Features**:
  1.	**Performance Comparison**: Evaluating the performance of OCR on images processed by DnCNN versus the original images. This helps determine the effectiveness of denoising in improving text extraction accuracy.
  
  2.	**Text Extraction Using Tesseract OCR**: The primary task is to extract text from images using Tesseract OCR. The images undergo preprocessing to improve OCR accuracy, including:
     
     - **Grayscale Conversion**: Simplifies the image by removing color information.
     - **Thresholding (Otsu's method)**: Binarizes the image to separate text from the background.
     - **Median Blur**: Reduces noise to enhance text clarity.




  
  3.	**Understandable Words Count**: The extracted text is analyzed to count the number of understandable words. This is done by:
     - Splitting the text into individual words.
     - Comparing each word against a list of English words from the NLTK corpus.
     - Returning the count of understandable words and the total number of words.


![image](https://github.com/bsabatino1/ML_PROJECT_773381_group2/assets/94707288/117c5ac7-be82-4fb4-8727-0e30f8e32020)

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


## [Section 4] Results

### Unsupervised model results
1. **Average Silhouette Score**: 0.64
   
   A silhouette score of 0.64 indicates a good clustering structure. This means that, on average, the samples are well-matched to their own clusters and clearly distinct from neighboring clusters.

3. **Davies-Bouldin Index**: 0.58
   
   A value of 0.58 indicates relatively low intra-cluster variance compared to inter-cluster variance. This means the clusters are compact and the separation between clusters is significant.

5. **Calinski-Harabasz Index**: 35593.91
   
   A high score of 35,593.91 suggests that the clusters are both compact and well-separated. This indicates that data points within each cluster are close to each other, while the clusters themselves are distinctly different from one another.


The silhouette score of 0.64, along with the low Davies-Bouldin Index of 0.58, indicates strong intra-cluster cohesion and distinct inter-cluster separation. The exceptionally high Calinski-Harabasz Index of 35,593.91 further confirms the effectiveness of the clustering solution, indicating that the clusters are well-defined and well-separated. Together, these results suggest that the unsupervised model using EfficientNetB0, KMeans, and UMAP is highly effective in creating meaningful and distinct clusters from the image data. 

The clusters have been established as follows:

**Cluster 1**: This cluster primarily comprises complex layouts, characterized by columns in the upper part and sections above.

**Cluster 2**: This cluster primarily consists of letter layouts, which are composed of a single block of text.

**Cluster 3**: This cluster predominantly contains biographies, organized into a few sections of text blocks.

**Cluster 4**: This cluster primarily features less dense layouts that do not adhere to a specific pattern.

**Cluster 5**: This cluster primarily includes standard curricula, generally consisting of two columns: the first column presents the section name, while the adjacent column displays the corresponding value.

It is important to note that, despite the formation of these defined clusters, there are outliers from other clusters within each one. This is due to the fact that not every layout is distinctly defined.




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

### Unsupervised Model with BERT and LayoutLM

**Current Status**

The script is designed to handle the task efficiently; however, the computational requirements exceed the available resources. As a result, the full results are not yet available. The initial tests have been conducted on a subset of 100 images, providing promising insights into the viability of the approach.

**Initial Results**
  - **Silhouette Score**: The initial clustering on 100 images yielded a silhouette score that indicates a reasonable separation between the clusters.
  - **Cluster Folders**: The resultant cluster folders from the initial test show a potential for accurately grouping similar images, aligning with the project's objectives.

**Challenges**
The primary challenge encountered is the high computational power required to process the entire dataset. Running the script for the complete dataset is computationally intensive and exceeds the capacity of the current environment.

At the moment of writing this report and the deadline, the script is in the process of running to completion. The full execution is expected to be completed a few hours after the deadline. The final results, including detailed analysis and updated metrics, will be provided as soon as the script finishes running.

Despite the current computational limitations, the initial tests on a subset of the images demonstrate that the approach has significant potential. The methodology, once fully executed, is expected to deliver the desired results effectively. We will update the report with the final results and analysis shortly after the script completes its run.

### OCR RESULTS

The OCR process generates a JSON file where the file name of the image serves as the key and the extracted text as the value. Despite enhancements made through the DnCNN to improve the quality of the scanned document, the extracted text remains incomplete due to the initial low quality of the scan.

<img width="400" alt="image" src="https://github.com/bsabatino1/ML_PROJECT_773381_group2/assets/94707288/d70a5239-adc0-4f88-a945-e166fdf4a72b">


## [Section 5] Conclusions

### Takeaway Points

1.	**Model Performance**:
    -	**Unsupervised Model**: Demonstrated high clustering effectiveness with a silhouette score of 0.64, a low Davies-Bouldin Index of 0.58, and a high Calinski-Harabasz Index of 35,593.91, indicating well-defined and distinct clusters.
    
    - **Semi-supervised Model**: Achieved a similar silhouette score of 0.64 but had a higher Davies-Bouldin Index of 5.01, reflecting potential issues with cluster compactness due to human labeling biases. The Calinski-Harabasz Index of 6,834.43 suggests good cluster definition but not as strong as the unsupervised model.
      
    - **BERT and LayoutLM Model**: Utilized advanced feature extraction methods, due to computational inefficiency it was not possible to complete the process.
    
2.	**Cluster Characteristics**:
   -	Identified distinct clusters in résumés, such as complex layouts, letter layouts, biographies, less dense layouts, and standard curricula. However, outliers were present, indicating that not all layouts were distinctly defined.


3.	**OCR and Denoising**:
   - OCR was enhanced by using the DnCNN model for image denoising, but the extracted text was still incomplete due to the initial low quality of the scanned documents.



### Future Work

1.	**Higher Computational Power**:
   - Investing in more computational resources to improve the performance and efficiency of feature extraction, dimensionality reduction, and clustering processes.

2.	**BERT Implementation on OCR**:
   - Implementing BERT for OCR to potentially enhance the accuracy and completeness of text extraction from scanned documents, leveraging BERT’s contextual text understanding capabilities.

3.	**Handling Human Biases**:
   - Developing strategies to mitigate human biases in the semi-supervised model’s manual labeling process to improve clustering accuracy and reduce high Davies-Bouldin Index values.


