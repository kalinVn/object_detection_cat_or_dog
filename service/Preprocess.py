import matplotlib
matplotlib.use("TkAgg")
import random
import os
import shutil

from config import SRC_FOLDER, TRAIN_SIZE, FILTER_SIZE, NUM_FILTERS, INPUT_SIZE, MAX_POOL_SIZE, BATCH_SIZE, STEPS_PER_EPOCH, EPOCHS
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from .Drive import Drive


class Preprocess:
    def __init__(self):
        self.training_data_generator = None
        self.testing_data_generator = None

    def train_test_split(self):
        Drive.clear_folders()
        Drive.create_folders()

    def create_test_training_data(self):
        training_data_generator = ImageDataGenerator(rescale=1/255)
        testing_data_generator = ImageDataGenerator(rescale=1/255)

        self.training_data_generator = training_data_generator.flow_from_directory(SRC_FOLDER + 'train/',
                                                                   target_size=(INPUT_SIZE, INPUT_SIZE),
                                                                   batch_size=BATCH_SIZE,
                                                                   class_mode='binary')

        self.testing_data_generator = testing_data_generator.flow_from_directory(SRC_FOLDER + 'test/',
                                                                  target_size=(INPUT_SIZE, INPUT_SIZE),
                                                                  batch_size=BATCH_SIZE,
                                                                  class_mode='binary')

    def get_training_data_generator(self):
        return self.training_data_generator

    def get_testing_data_generator(self):
        return self.testing_data_generator

    def get_images(self):
        return self.images


