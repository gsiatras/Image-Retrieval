# Image Retrieval System
## Overview
The Image Retrieval System is a project designed for efficient image retrieval using a Siamese model with triplet loss and ResNet50 as the backbone. The system is trained on the Google Landmarks dataset and evaluated on a custom dataset, providing a powerful tool for similarity-based image search.

## Components
### 1. DatasetHandler
The DatasetHandler module is responsible for managing datasets, including the test set and evaluation set. It ensures seamless loading and handling of image data, facilitating the training and evaluation processes.

### 2. Model
The core of the system is the Siamese model, which leverages triplet loss for training. The backbone architecture is built on the pre-trained ResNet50 model, known for its strong feature extraction capabilities. The model is designed to learn a robust representation of images, enabling effective similarity comparisons.

### 3. Training Data
The Siamese model is trained on the Google Landmarks dataset, which can be accessed here. This dataset provides a diverse collection of images from various landmarks, allowing the model to learn generalized features.      
Download dataset from: https://www.kaggle.com/datasets/mattbast/google-landmarks-2020-tfrecords, add it to a folder name Dataset in the working dir.

### 4. Evaluation
The trained model is evaluated on a custom dataset from Google Landmarks containing 100 images, 3 of them are similar to the query.
Download dataset from: [https://www.dropbox.com/scl/fo/v2cnpxvp50u43q8xs4ohg/h?rlkey=e6dhfxoe2bola2c7amd2k735m&dl=0](https://www.dropbox.com/scl/fo/v2cnpxvp50u43q8xs4ohg/h?rlkey=93l99utkrdpjx2x99jl0zgva1&dl=0), add it to a folder name Test_set in the working dir.  

### 5. GEM Pooling Layer
The system incorporates the GEM (Generalized Mean) pooling layer, enhancing the model's ability to capture global context information. GEM pooling is known for its effectiveness in aggregating features, contributing to improved performance in image retrieval tasks.

### 6. Faiss Indexing
To optimize and accelerate the image retrieval process, the system utilizes Faiss indexing. Faiss is a library for efficient similarity search and clustering of dense vectors, providing a robust solution for large-scale retrieval scenarios.

## Results
### Training Loss:
![loss](https://github.com/gsiatras/Image_Retrieval_System/assets/94067900/6496cacf-10ce-4665-88d2-c46c1177b5f3)
### Retrieved images:    
![60ep](https://github.com/gsiatras/Image_Retrieval_System/assets/94067900/b4367399-87c4-4e87-8b24-0f971ae4a955)
<img width="501" alt="Screenshot_2" src="https://github.com/gsiatras/Image_Retrieval_System/assets/94067900/4158799f-6861-4986-b2a4-c9fb87b1dea2">



