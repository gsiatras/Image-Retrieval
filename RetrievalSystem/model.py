from dataset_handler import DatasetHandler
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from scipy.spatial import distance
from tensorflow.keras.callbacks import Callback
import numpy as np




class SiameseModel:
    def __init__(self, image_size=128, feature_dim=2048, learning_rate=0.000001):
        self.input_shape = (image_size, image_size, 3)
        self.learning_rate = learning_rate
        self.feature_dim = feature_dim
        self.reg = tf.keras.regularizers
        self.base_model = self._build_base_model()
        self.triplet_loss = TripletLoss(margin=0.1)
        self.model, self.emdeding_model = self._build_model()

        self.callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True),
        ]

    def _build_base_model(self):
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        base_model.trainable = False
        return base_model

    def _build_model(self):
        # Define the base model
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        base_model.trainable = False
        xavier_initializer = tf.keras.initializers.GlorotNormal()


        # Define input layers
        x_input = tf.keras.layers.Input(shape=self.input_shape)
        anchor_input = tf.keras.layers.Input(shape=self.input_shape, name='anchor_input')
        positive_input = tf.keras.layers.Input(shape=self.input_shape, name='positive_input')
        negative_input = tf.keras.layers.Input(shape=self.input_shape, name='negative_input')

        # Flatten and add Gem pooling layer
        gem = GeMPoolingLayer()
        dense_layer = layers.Dense(self.feature_dim, activation='softplus', kernel_regularizer=self.reg.l2(),
                                   dtype='float32', kernel_initializer=xavier_initializer)

        # model structure
        x = base_model(x_input)
        x = gem(x)
        x = dense_layer(x)

        # evaluation model
        embedding_model = tf.keras.models.Model(inputs=x_input, outputs=x, name="embedding")

        # Build Siamese model with shared base_model
        anchor_output = embedding_model(anchor_input)
        positive_output = embedding_model(positive_input)
        negative_output = embedding_model(negative_input)

        # Create the Siamese model
        model = tf.keras.models.Model(
            inputs=[anchor_input, positive_input, negative_input],
            outputs=[anchor_output, positive_output, negative_output]
        )

        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.000001, beta_1=0.9)
        model.compile(optimizer=optimizer, loss=self.triplet_loss)
        return model, embedding_model



    def fit(self, data_generator, epochs, steps_per_epoch, eval_test_data, train_test_data):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        testing_callback = TestingCallback(self.emdeding_model, eval_test_data, train_test_data)
        history = self.model.fit(
            data_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[testing_callback] + self.callbacks,
        )
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

        return history

    def save_weights(self, filepath):
        """
        Save the weights of the trainable layers in the Siamese model.

        Parameters:
            - filepath (str): Full filepath to save the weights.
        """
        # Extract the directory and filename from the provided filepath
        directory, filename = os.path.split(filepath)

        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Save the weights to the specified filepath
        self.model.save_weights(filepath)


    def load_weights(self, filepath):
        """
        Load the weights into the trainable layers of the Siamese model.

        Parameters:
            - filepath (str): Filepath to load the weights from.
        """
        self.model.load_weights(filepath)

    def calclulate_distance(self, vector1, vector2):
        return tf.reduce_sum(tf.square(tf.subtract(vector1, vector2)), axis=-1)


    def distance_test(self, anchors, positives, negatives):
        '''
        :param anchors:
        :param positives:
        :param negatives:
        :return: positive and negative distances
        '''
        pos_dist = []
        neg_dist = []

        anchor_encodings = self.emdeding_model.predict(anchors)
        positive_encodings = self.emdeding_model.predict(positives)
        negative_encodings = self.emdeding_model.predict(negatives)

        for i in range(len(anchors)):
            pos_dist.append(
                distance.euclidean(anchor_encodings[i], positive_encodings[i])
            )

            neg_dist.append(
                distance.euclidean(anchor_encodings[i], negative_encodings[i])
            )

        return pos_dist, neg_dist


class TestingCallback(Callback):
    '''
    A call back to test the network after each epoch
    '''
    def __init__(self, model, eval_test_data, train_test_data):
        super().__init__()
        self.model = model
        self.eval_test_data = eval_test_data
        self.train_test_data = train_test_data

    def on_epoch_end(self, epoch, logs=None):
        # open eval and train data
        for images, landmark_id in self.train_test_data:
            train_anchors = images['anchor_input']
            train_positives = images['positive_input']
            train_negatives = images['negative_input']

        for images, landmark_id in self.eval_test_data:
            eval_anchors = images['anchor_input']
            eval_positives = images['positive_input']
            eval_negatives = images['negative_input']

        # Get the embeddings for all the data for train set
        train_anchor_embedding, train_positive_embedding, train_negative_embedding  = self.model.predict([train_anchors, train_positives, train_negatives])

       # Get the embeddings for all the data for eval set
        eval_anchor_embedding, eval_positive_embedding, eval_negative_embedding = self.model.predict([eval_anchors, eval_positives, eval_negatives])

        # Distances for train and eval set
        train_pos_dist = []
        train_neg_dist = []

        eval_pos_dist = []
        eval_neg_dist = []

        for i in range(len(train_anchors)):
            train_pos_dist.append(
                distance.euclidean(train_anchor_embedding[i], train_positive_embedding[i])
            )

            train_neg_dist.append(
                distance.euclidean(train_anchor_embedding[i], train_negative_embedding[i])
            )

        for i in range(len(eval_anchors)):
            eval_pos_dist.append(
                distance.euclidean(eval_anchor_embedding[i], eval_positive_embedding[i])
            )

            eval_neg_dist.append(
                distance.euclidean(eval_anchor_embedding[i], eval_negative_embedding[i])
            )

        # Do something with the distances, e.g., print or store them
        print(f"Epoch {epoch + 1} Train Positive Mean Distance:", np.mean(train_pos_dist))
        print(f"Epoch {epoch + 1} Train Negative Mean Distance:", np.mean(train_neg_dist))
        print(f"Epoch {epoch + 1} Eval Positive Mean Distance:", np.mean(eval_pos_dist))
        print(f"Epoch {epoch + 1} Eval Negative Mean Distance:", np.mean(eval_neg_dist))



class TripletLoss(tf.keras.losses.Loss):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__(**kwargs)
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return tf.reduce_sum(tf.square(tf.subtract(x1, x2)), axis=-1)

    def call(self, y_true, y_pred,):
        anchors = y_pred[0]
        positives = y_pred[1]
        negatives = y_pred[2]
        # y_true is not used because TripletLoss is unsupervised
        distance_positive = self.calc_euclidean(anchors, positives)
        distance_negative = self.calc_euclidean(anchors, negatives)

        losses = tf.add(tf.subtract(distance_positive, distance_negative), self.margin)
        loss = tf.reduce_sum(tf.maximum(losses, 0))
        return loss


class GeMPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, p=1., eps=1e-6):
        super().__init__()
        self.p = p
        self.eps = eps

    def call(self, inputs: tf.Tensor, **kwargs):
        inputs = tf.clip_by_value(
            inputs,
            clip_value_min=self.eps,
            clip_value_max=tf.reduce_max(inputs)
        )
        inputs = tf.pow(inputs, self.p)
        inputs = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        inputs = tf.pow(inputs, 1. / self.p)

        return inputs

    def get_config(self):
        return {
            'p': self.p,
            'eps': self.eps
        }




