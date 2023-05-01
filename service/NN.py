from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense

from .Preprocess import Preprocess
from factory import Model as ModelFactory
from config import STEPS_PER_EPOCH, EPOCHS


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
        self.model = self.model_factory.get_keras_model('VGG16')

        self.preprocess.train_test_split()

        self.preprocess.create_test_training_data()
        self.x_train = self.preprocess.get_training_data_generator()
        self.x_test = self.preprocess.get_testing_data_generator()

        self.model.fit_generator(self.x_train, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, verbose=1)

    def score(self):
        score = self.model.evaluate_generator(self.x_test, steps=100)

        for idx, metric in enumerate(self.model.metrics_names):
            print("{}: {}".format(metric, score[idx]))