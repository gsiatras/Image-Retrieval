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

### 5. GEM Pooling Layer
The system incorporates the GEM (Generalized Mean) pooling layer, enhancing the model's ability to capture global context information. GEM pooling is known for its effectiveness in aggregating features, contributing to improved performance in image retrieval tasks.

### 6. Faiss Indexing
To optimize and accelerate the image retrieval process, the system utilizes Faiss indexing. Faiss is a library for efficient similarity search and clustering of dense vectors, providing a robust solution for large-scale retrieval scenarios.
