import random
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

from config import SRC_FOLDER, FILE_TO_REMOVE_CATS, FILE_TO_REMOVE_DOGS, TRAIN_SIZE


class Drive:

    @staticmethod
    def clear_folders():
        shutil.rmtree(SRC_FOLDER + 'train/Cat/', ignore_errors=True)
        shutil.rmtree(SRC_FOLDER + 'train/Dog', ignore_errors=True)
        shutil.rmtree(SRC_FOLDER + 'test/Cat', ignore_errors=True)
        shutil.rmtree(SRC_FOLDER + 'test/Dog', ignore_errors=True)

    @staticmethod
    def create_folders():
        os.makedirs(SRC_FOLDER + 'train/Cat')
        os.makedirs(SRC_FOLDER + 'train/Dog')
        os.makedirs(SRC_FOLDER + 'test/Cat')
        os.makedirs(SRC_FOLDER + 'test/Dog')

        cat_class = 'cat'
        dog_class = 'dog'
        cats_folder = Drive.get_folder_by_object_class(cat_class)
        cats_files_to_remove = Drive.get_files_to_remove_by_object_class(cat_class)
        Drive.clear_images(cats_folder, cats_files_to_remove)

        dogs_folder = Drive.get_folder_by_object_class(dog_class)
        dogs_files_to_remove = Drive.get_files_to_remove_by_object_class(dog_class)

        Drive.clear_images(dogs_folder, dogs_files_to_remove)

        Drive.add_images(cats_folder)
        Drive.add_images(dogs_folder)

    @staticmethod
    def clear_images(folder, files_to_remove):
        _, _, images = next(os.walk(SRC_FOLDER + folder))

        for file in files_to_remove:
            images.remove(file)

    @staticmethod
    def add_images(folder):
        _, _, images = next(os.walk(SRC_FOLDER + folder))

        images_length = len(images)
        images_train_length = int(images_length * TRAIN_SIZE)
        images_test_length = images_length - images_train_length

        images_train = random.sample(images, images_train_length)
        images_test = [img for img in images if img not in images_train]

        train_folder_name = 'train/'
        test_folder_name = 'test/'
        for img in images:
            src_img_path = SRC_FOLDER + folder + img

            dist_img_path = SRC_FOLDER + train_folder_name + folder
            shutil.copy(src=src_img_path, dst=dist_img_path)

        for img in images_test:
            src_img_path = SRC_FOLDER + folder + img
            dist_img_path = SRC_FOLDER + test_folder_name + folder
            shutil.copy(src=src_img_path, dst=dist_img_path)

    @staticmethod
    def get_folder_by_object_class(object_class):
        folder_name = 'Dog/'

        if object_class == "cat":
            folder_name = 'Cat/'

        return folder_name

    @staticmethod
    def get_files_to_remove_by_object_class(object_class):
        files_to_removed = FILE_TO_REMOVE_DOGS

        if object_class == "cat":
            files_to_removed = FILE_TO_REMOVE_CATS

        return files_to_removed

    @staticmethod
    def get_images_augmentation(class_obj):
        images = []
        images_path = 'dataset/PetImages/Dog/'
        if class_obj == 'cat':
            images_path = 'dataset/PetImages/Cat/'

        img_generator = ImageDataGenerator(rotation_range=30,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True,
                                            fill_mode='nearest')

        _, _, current_images = next(os.walk(images_path))
        random_img = random.sample(current_images, 1)[0]

        random_img = plt.imread(images_path + random_img)
        images.append(random_img)
        random_img = random_img.reshape((1,) + random_img.shape)
        simple_augmented_images = img_generator.flow(random_img)

        for _ in range(5):
            augment_images = simple_augmented_images.next()
            for img in augment_images:
                images.append(img.astype('uint8'))

        return images


