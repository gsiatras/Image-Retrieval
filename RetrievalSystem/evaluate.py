from dataset_handler import DatasetHandler
from model import SiameseModel
import matplotlib.pyplot as plt
import os
from scipy.spatial import distance
import faiss
import matplotlib.image as mpimg
import numpy as np
import warnings
import tensorflow as tf
from scipy.spatial import distance


batch_size = 64
epochs = 20
steps_per_epoch = 140933 // batch_size
rate = 0.0000001

image_size = 128
embed_size = 2048
smodel = SiameseModel()
dt = DatasetHandler("Dataset")


def test_system():
    '''
    Test the system on the evaluating set
    :return: 16 similar images to the query
    '''
    index = faiss.IndexFlatL2(smodel.feature_dim)

    images = dt.get_test_images(os.path.join("..", "Test_set"))
    #query_img1 = images[0]
    query_img = np.expand_dims(images[0], axis=0)
    new_images = images[1:]

    query_representation = smodel.emdeding_model.predict(query_img)

    batch_embeddings = smodel.emdeding_model.predict(new_images)
    index.add(np.array(batch_embeddings))


    faiss.write_index(index, "faiss_index.index")

    k = 16  # Number of nearest neighbors to retrieve
    distances, indices = index.search(query_representation, k)


    similar_image_paths = [new_images[i] for i in indices[0]]

    # Plot the query image

    plt.figure(figsize=(6, 6))  # Adjust the figure size as needed
    plt.imshow(images[0])
    plt.title('Query Image')
    plt.axis('off')
    plt.show()

    # Plot the retrieved similar images in a 3x3 grid
    plt.figure(figsize=(10, 10))  # Adjust the figure size as needed

    for i in range(min(len(new_images), 16)):
        plt.subplot(4, 4, i + 1)  # Adjust the subplot indices as needed
        plt.imshow(similar_image_paths[i])
        plt.title(f'Distance: {distances[0][i]:.6f}')
        plt.axis('off')

    plt.show()


if __name__ == '__main__':
    smodel.load_weights(os.path.join("..", "Model", f"siamese_model_weights(60epochs_augment_small_128.h5"))
    test_system()