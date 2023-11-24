import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import cv2

class DatasetHandler(object):

    def __init__(self, dataset_folder, image_size=128, batch_size=64):
        self.dataset_folder = dataset_folder
        self.setup_pipeline()
        self.setup_testing_pipeline()

        self.image_size = image_size
        self.batch_size = batch_size


    def get_test_images(self, folder_path):
        """
        Retrieve evaluating images to
        :param folder_path:path the folder containing the images
        :return: numpy array containing the images
        """
        # Get a list of image files in the folder
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

        # Initialize an empty list to store decoded images
        decoded_images = []

        # Loop through each image file and decode it
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            decoded_image = self.decode_image2(image_path)
            decoded_images.append(decoded_image)

        # Convert the list of images to a NumPy array
        image_array = np.array(decoded_images)

        return image_array


    def setup_pipeline(self):
        '''
        setup the training pipeline
        :return:
        '''
        # getting a list of the tfrecord filenames
        filenames = [os.path.join("..", "Dataset", f"train{i:02d}.tfrec") for i in range(12)]

        # begin the pipeline by telling it to expect tfrecords
        self.train_data = tf.data.TFRecordDataset(
            filenames,
            num_parallel_reads=tf.data.experimental.AUTOTUNE
        )

        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        self.train_data = self.train_data.with_options(ignore_order)

        """
            Request a TPU for the notebook.
            If there isn't one (if the notebook was not configured to use TPU) it will get a CPU or GPU instead.
        """
        self.gpu = tf.config.list_physical_devices('GPU')
        if self.gpu:
            print('Running on GPU:', self.gpu)
        else:
            print('No GPU available, running on CPU.')




    def setup_testing_pipeline(self):
        '''
        Setup the testing pipeline
        :return:
        '''
        self.test_data = tf.data.TFRecordDataset(
            os.path.join("..", "Dataset", f"train14.tfrec"),
            num_parallel_reads=tf.data.experimental.AUTOTUNE
        )

    def find_steps(self):
        '''
        Find how many images in each subdataset
        :return:
        '''
        # Create a TFRecordDataset
        dataset = tf.data.TFRecordDataset(
            os.path.join("..", "Dataset", f"train01.tfrec"),
            num_parallel_reads=tf.data.experimental.AUTOTUNE
        )

        parsed_dataset = dataset.map(self.get_triplet)

        num_steps = 0
        for _ in parsed_dataset:
            num_steps += 1

        print(f'The dataset contains {num_steps} steps.')


    def get_triplet(self, example):
        """
        Input: example from a tfrecord file
        Get a triplet of data
        :return: a triplet of data
        """
        tfrec_format = {
            "anchor_img": tf.io.FixedLenFeature([], tf.string),
            "positive_img": tf.io.FixedLenFeature([], tf.string),
            "negative_img": tf.io.FixedLenFeature([], tf.string),
        }

        example = tf.io.parse_single_example(example, tfrec_format)

        x = {
            'anchor_input': self.decode_image(example['anchor_img']),
            'positive_input': self.decode_image(example['positive_img']),
            'negative_input': self.decode_image(example['negative_img']),
        }

        return x, [0, 0, 0]


    def decode_image(self, image_data):
        """
        Decode an image from the data set
        then augment it and return it
        :return: Decoded image
        """
        image = tf.image.decode_jpeg(image_data, channels=3)
        image = tf.cast(image, tf.float32) / 255.
        image = tf.image.resize(image, (self.image_size, self.image_size), method='nearest')

        image = self.augment(image)

        return image

    def decode_image2(self, image):
        """
        Decode an image from the data set
        then augment it and return it
        :return: Decoded image
        """
        #image = Image.open(image)
        image = cv2.imread(image)
        # image = tf.image.decode_jpeg(image, channels=3)
        # image = image.resize((self.image_size, self.image_size))
        image = tf.cast(image, tf.float32) / 255.
        image = tf.image.resize(image, (self.image_size, self.image_size), method='nearest')
        #image = np.asarray(image) / 255.0
        #print(image.shape)

        # image = self.augment(image)

        return image


    def augment(self, image):
        """
        Augment the given image to add further variety to the dataset.
        :return: Augmented image
        """
        rand_aug = np.random.choice([0, 1, 2, 3])

        if rand_aug == 0:
            image = tf.image.random_brightness(image, max_delta=0.4)
        elif rand_aug == 1:
            image = tf.image.random_contrast(image, lower=0.2, upper=0.5)
        elif rand_aug == 2:
            image = tf.image.random_hue(image, max_delta=0.2)
        else:
            image = tf.image.random_saturation(image, lower=0.2, upper=0.5)

        rand_aug = np.random.choice([0, 1, 2, 3])

        if rand_aug == 0:
            image = tf.image.random_flip_left_right(image)
        elif rand_aug == 1:
            image = tf.image.random_flip_up_down(image)
        elif rand_aug == 2:
            rand_rot = np.random.randn() * 45
            image = tfa.image.rotate(image, rand_rot)
        else:
            image = tfa.image.transform(image, [1.0, 1.0, -50, 0.0, 1.0, 0.0, 0.0, 0.0])

        image = tf.image.random_crop(image, size=[100, 100, 3])
        image = tf.image.resize(image, (self.image_size, self.image_size))

        return image


    def get_data(self):
        '''
        Get the pipeline to feed the network
        :return:pipe
        '''
        self.train_data = self.train_data.map(
            self.get_triplet,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        #
        self.train_data = self.train_data.repeat() # Repeat enables us to train for more than one epoch
        self.train_data = self.train_data.shuffle(1024) # Shuffle helps to prevent overfit
        self.train_data = self.train_data.batch(self.batch_size) # Batching ensures the right amount of data gets put into the model per step
        self.train_data = self.train_data.prefetch(tf.data.experimental.AUTOTUNE) # Prefetch gets the next batch of data while the model is training on the previous batch
        return self.train_data


    def get_test_data(self):
        '''
        Get the testing data to test the network
        :return:
        '''
        self.test_data = self.test_data.map(
            self.get_triplet,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.test_data = self.test_data.repeat()
        self.test_data = self.test_data.shuffle(1024)
        self.test_data = self.test_data.batch(self.batch_size)
        self.test_data = self.test_data.prefetch(tf.data.experimental.AUTOTUNE)
        return self.test_data



    def example_images(self):
        '''
        :return: A batch of training data
        '''
        for images, landmark_id in self.train_data.take(1):
            anchors = images['anchor_input']
            positives = images['positive_input']
            negatives = images['negative_input']

        #     for i in range(5):
        #         axes[i, 0].set_title('Anchor')
        #         axes[i, 0].imshow(anchors[i])
        #
        #         axes[i, 1].set_title('Positive')
        #         axes[i, 1].imshow(positives[i])
        #
        #         axes[i, 2].set_title('Negative')
        #         axes[i, 2].imshow(negatives[i])
        # plt.show()
        return anchors, positives, negatives


#
# if __name__ == '__main__':
#     dt = DatasetHandler("Dataset")
#     dt.find_steps()