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

def train():
    '''
    Training the network
    :return:
    '''
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

    dt.get_test_data()
    history = smodel.fit(
        dt.get_data(),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        eval_test_data=dt.test_data.take(1),
        train_test_data=dt.train_data.take(1),
    )
    print(history.history)

    plt.title('Model loss')
    plt.plot(history.history['loss'])
    smodel.save_weights(os.path.join("..", "Model", f"siamese_model_weights(60epochs_augment_small_128.h5"))

    warnings.resetwarnings()

def test_on_train_set():
    '''
    Test the network on the training set
    :return: 5 triplets with distances
    '''
    dt.get_data()
    anchors, positives, negatives = dt.example_images()

    """Test multuple distances"""
    pos_dist, neg_dist = smodel.distance_test(anchors[0:5], positives[0:5], negatives[0:5])
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))

    for i in range(5):
        axes[i, 0].set_title('Anchor')
        axes[i, 0].imshow(anchors[i])

        axes[i, 1].set_title('Positive dist: {:.2f}'.format(pos_dist[i]))
        axes[i, 1].imshow(positives[i])

        axes[i, 2].set_title('Negative dist: {:.2f}'.format(neg_dist[i]))
        axes[i, 2].imshow(negatives[i])

    plt.show()



if __name__ == '__main__':
    smodel.model.summary()
    smodel.load_weights(os.path.join("..", "Model", f"siamese_model_weights(60epochs_augment_small_128.h5"))
    train()
    test_on_train_set()










