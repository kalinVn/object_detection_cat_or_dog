import os.path

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
import tensorflow as tf

from .Preprocess import Preprocess
from .Drive import Drive
from factory import Model as ModelFactory
from config import STEPS_PER_EPOCH, EPOCHS, SRC_FOLDER

import cv2

import numpy as np

class NN:

    def __init__(self, model_type='cnn'):
        self.conv_model_2D_1 = None
        self.conv_model_2D_2 = None
        self.model = None
        self.x_test = None
        self.x_train = None
        self.model_type = model_type
        self.preprocess = Preprocess()
        self.model_factory = ModelFactory()

    def set_x_train(self, x_train):
        self.x_train = x_train

    def set_x_test(self, x_test):
        self.x_test = x_test

    def fit(self):

        # use conv nn
        # self.model = self.model_factory.get_keras_model('VGG16')
        # if not ( os.path.isdir(SRC_FOLDER + 'train/Cat') == True and os.path.isdir(SRC_FOLDER + 'train/Dog') == True\
        #         and os.path.isdir(SRC_FOLDER + 'test/Dog') == True and os.path.isdir(SRC_FOLDER + 'test/Cat') == True):
        #
        #     self.preprocess.init_folders()
        #
        # self.preprocess.create_test_training_data()
        # self.x_train = self.preprocess.get_training_data_generator()
        # self.x_test = self.preprocess.get_testing_data_generator()
        # x = np.concatenate([self.x_test.next()[0] for i in range(self.x_test.__len__())])
        #
        # self.model.fit_generator(self.x_train, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, verbose=1)


        # use pretrained model
        x_train, x_test, y_train, y_test = Drive.train_test_split()

        # x_train_scaled, x_test_scaled = self.preprocess.scale_pixels(x_train, x_test)

        self.model = self.model_factory.get_keras_model('mobile_net_v2')
        print(os.path.exists('store/models/model_SaveModel_format_test'))

        if (os.path.exists('store/models/model_SaveModel_format_test')):
            self.model = tf.keras.models.load_model("store/models/model_SaveModel_format_test");
        else:
            self.model.fit(x_train, y_train, epochs=5, verbose=1)
            self.model.save('store/models/model_SaveModel_format_test')

        score, acc = self.model.evaluate(x_test, y_test)
        print('Test Loss = ', score)
        print('Test Accuracy = ', acc)


    def score(self):
        score = self.model.evaluate_generator(self.x_test, steps=100)

        for idx, metric in enumerate(self.model.metrics_names):
            print("{}: {}".format(metric, score[idx]))

    def predict(self, path):
        img = cv2.imread(path)
        img_resized = cv2.resize(img, (224, 224))
        img_reshaped = np.reshape(img_resized, [1, 224, 224, 3])

        prediction = self.model.predict(img_reshaped)
        print(prediction)
        input_predict_label = np.argmax(prediction)

        if input_predict_label == 1:
            print("The image represent a Cat")
        else:
            print("The image represent a Dog")