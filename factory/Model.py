from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.models import Model as VGG16Model
from keras.layers import Dropout, Flatten, Dense
from config import FILTER_SIZE, NUM_FILTERS, INPUT_SIZE, MAX_POOL_SIZE, STEPS_PER_EPOCH, EPOCHS

import tensorflow as tf
import tensorflow_hub as hub

class Model:

    @staticmethod
    def get_keras_model(model_name):
        print(model_name)
        if model_name == 'cnn':
            model = Sequential()
            conv_model_2D_1 = Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE),
                                          input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
                                          activation="relu")
            conv_model_2D_2 = Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE), activation="relu")
            model.add(conv_model_2D_1)
            model.add(MaxPooling2D(pool_size=(MAX_POOL_SIZE, MAX_POOL_SIZE)))
            model.add(conv_model_2D_2)
            model.add(MaxPooling2D(pool_size=(MAX_POOL_SIZE, MAX_POOL_SIZE)))
            model.add(Flatten())
            model.add(Dense(units=128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(units=1, activation='sigmoid'))
            model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model
        elif model_name == 'mobile_net_v2':
            mobile_net_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
            pretrained_model = hub.KerasLayer(mobile_net_url, input_shape=(224, 224, 3), trainable=False)
            num_of_classes = 2
            model = tf.keras.Sequential([
                pretrained_model,
                tf.keras.layers.Dense(num_of_classes)
            ])

            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['acc']
            )

            return model

        vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(INPUT_SIZE, INPUT_SIZE, 3))

        for layer in vgg16.layers:
            layer.trainable = False

        input_ = vgg16.input

        output_ = vgg16(input_)
        last_layer = Flatten(name='flatten')(output_)
        last_layer = Dense(1, activation='sigmoid')(last_layer)
        model_vgg16 = VGG16Model(input_, last_layer)

        model_vgg16.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model_vgg16
