# Image Retrieval System
## Overview
The Image Retrieval System is a tool designed for efficient image retrieval, leveraging a Siamese model with triplet loss and ResNet50 as the backbone. Indexing using faiss. This project is implemented in TensorFlow.

## Components
### 1. DatasetHandler
The DatasetHandler module is responsible for managing datasets, including the test set and evaluation set. It ensures seamless loading and handling of image data, facilitating the training and evaluation processes.

### 2. Model
The core of the system is the Siamese model, which leverages triplet loss for training. The backbone architecture is built on the pre-trained ResNet50 model, known for its strong feature extraction capabilities. The model is designed to learn a robust representation of images, enabling effective similarity comparisons.

### 3. Training Data
The Siamese model is trained on the Google Landmarks dataset, which can be accessed here. This dataset provides a diverse collection of images from various landmarks, allowing the model to learn generalized features.      
Download dataset from: https://www.kaggle.com/datasets/mattbast/google-landmarks-2020-tfrecords, add it to a folder name Dataset in the working dir.

### 4. Evaluation
The trained model is evaluated on a custom dataset from Google Landmarks containing 100 images, 5 of the same monument, the first is used as the query.
Download dataset from: [https://www.dropbox.com/scl/fo/v2cnpxvp50u43q8xs4ohg/h?rlkey=e6dhfxoe2bola2c7amd2k735m&dl=0](https://www.dropbox.com/scl/fo/v2cnpxvp50u43q8xs4ohg/h?rlkey=93l99utkrdpjx2x99jl0zgva1&dl=0), add it to a folder name Test_set in the working dir.  

### 5. GEM Pooling Layer
The system incorporates the GEM (Generalized Mean) pooling layer, enhancing the model's ability to capture global context information. GEM pooling is known for its effectiveness in aggregating features, contributing to improved performance in image retrieval tasks.

### 6. Faiss Indexing
To optimize and accelerate the image retrieval process, the system utilizes Faiss indexing. Faiss is a library for efficient similarity search and clustering of dense vectors, providing a robust solution for large-scale retrieval scenarios.

## Results
### Training Loss:
![loss](https://github.com/gsiatras/Image_Retrieval_System/assets/94067900/6496cacf-10ce-4665-88d2-c46c1177b5f3)
### Query image:
![query](https://github.com/gsiatras/Image_Retrieval_System/assets/94067900/74cab87e-3c2b-4f6a-b7ef-6cd59c63cc9f)
### Retrieved images:    
<img width="501" alt="Screenshot_2" src="https://github.com/gsiatras/Image_Retrieval_System/assets/94067900/8f7dfc7a-79f4-4275-8661-b5b507708ce5">        

We succesfully retrieve all 4 images of the monument on our top 16. However, we have big distances.

## Usage
1.Clone the repository      
2.Install dependencies:      
pip install -r requirements.txt
3. Download the datasets and place in the appropriate directory described above   
4. Download the weights if needed from https://www.dropbox.com/scl/fi/3fapxvmh0aah1iscim65n/siamese_model_weights-60epochs_augment_small_128.h5?rlkey=k1i4v3ud6f3suhqb1vw0ivow6&dl=0       
5. Train the Siamese model:      
python train.py
6. Evaluate the model on your custom dataset:      
python evaluate.py






