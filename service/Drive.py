import random
import os
import shutil

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from PIL import Image
from config import SRC_FOLDER, FILE_TO_REMOVE_CATS, FILE_TO_REMOVE_DOGS, TRAIN_SIZE
import cv2
import glob



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
        images_path = 'dataset/PetImages/test/Dog/'
        if class_obj == 'cat':
            images_path = 'dataset/PetImages/test/Cat/'

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

    @staticmethod
    def list_dir_fileNames(dir):
        names = os.listdir(dir)

        for name in names:
            print(names)

    @staticmethod
    def resize_images():
        train_original_folder_path_cat = SRC_FOLDER + '/train/Cat/'
        train_resized_folder_path_cat = SRC_FOLDER + '/train/Cat_Resized/'
        train_original_folder_path_dog = SRC_FOLDER + '/train/Dog/'
        train_resized_folder_path_dog = SRC_FOLDER + '/train/Dog_Resized/'
        train_items_count_cat = len(os.listdir(SRC_FOLDER + 'train/Cat'))
        train_items_count_dog = len(os.listdir(SRC_FOLDER + 'train/Dog'))
        Drive.create_resizied_folder(train_original_folder_path_cat, train_resized_folder_path_cat, train_items_count_cat)
        Drive.create_resizied_folder(train_original_folder_path_dog, train_resized_folder_path_dog, train_items_count_dog)

        test_original_folder_path_cat = SRC_FOLDER + '/test/Cat/'
        test_resized_folder_path_cat = SRC_FOLDER + '/test/Cat_Resized/'
        test_original_folder_path_dog = SRC_FOLDER + '/test/Dog/'
        test_resized_folder_path_dog = SRC_FOLDER + '/test/Dog_Resized/'
        test_items_count_cat = len(os.listdir(SRC_FOLDER + 'test/Cat'))
        test_items_count_dog = len(os.listdir(SRC_FOLDER + 'test/Dog'))
        Drive.create_resizied_folder(test_original_folder_path_cat, test_resized_folder_path_cat,
                                     test_items_count_cat)
        Drive.create_resizied_folder(test_original_folder_path_dog, test_resized_folder_path_dog,
                                     test_items_count_dog)

        # if not os.path.isdir(resized_folder_path_cat):
        #     os.mkdir(resized_folder_path_cat)
        #     original_folder_cat = os.listdir(original_folder_path_cat)
        #     for i in range(2000):
        #         file_name_cat = original_folder_cat[i]
        #         img_path_cat = original_folder_path_cat + file_name_cat
        #         img_cat = Image.open(img_path_cat)
        #         img_cat = img_cat.resize((224, 224))
        #         img_cat = img_cat.convert('RGB')
        #
        #         new_img_path_cat = resized_folder_path_cat + file_name_cat
        #         img_cat.save(new_img_path_cat)

    @staticmethod
    def create_resizied_folder(original_folder_path, resized_folder_path, items_count):
        if not os.path.isdir(resized_folder_path):
            os.mkdir(resized_folder_path)
            original_folder_cat = os.listdir(original_folder_path)

            for i in range(items_count):
                file_name_cat = original_folder_cat[i]
                img_path_cat = original_folder_path + file_name_cat
                img_cat = Image.open(img_path_cat)
                img_cat = img_cat.resize((224, 224))
                img_cat = img_cat.convert('RGB')

                new_img_path_cat = resized_folder_path + file_name_cat
                img_cat.save(new_img_path_cat)

    @staticmethod
    def train_test_split():
        files_cat_train = []
        files_dog_train = []
        train_resized_folder_path_cat = SRC_FOLDER + 'train/Cat_Resized/'
        train_resized_folder_path_dog = SRC_FOLDER + 'train/Dog_Resized/'

        files_cat_train.extend(glob.glob(train_resized_folder_path_cat + '*.jpg'))
        files_dog_train.extend(glob.glob(train_resized_folder_path_dog + '*.jpg'))
        train_cat_images_as_np_array = np.asarray([cv2.imread(file) for file in files_cat_train])
        train_dog_images_as_np_array = np.asarray([cv2.imread(file) for file in files_dog_train])
        # print(cat_images_as_np_array.shape)
        train_cat_images_labels = [{'label': 0, 'src': file} for file in train_cat_images_as_np_array]
        train_dog_images_labels = [{'label': 1, 'src': file} for file in train_dog_images_as_np_array]

        train_data_labels_and_src = train_cat_images_labels + train_dog_images_labels
        np.random.shuffle(train_data_labels_and_src)

        y_train = [item['label'] for item in train_data_labels_and_src]
        x_train = [item['src'] for item in train_data_labels_and_src]

        files_cat_test = []
        files_dog_test = []
        test_resized_folder_path_cat = SRC_FOLDER + 'test/Cat_Resized/'
        test_resized_folder_path_dog = SRC_FOLDER + 'test/Dog_Resized/'

        files_cat_test.extend(glob.glob(test_resized_folder_path_cat + '*.jpg'))
        files_dog_test.extend(glob.glob(test_resized_folder_path_dog + '*.jpg'))
        test_cat_images_as_np_array = np.asarray([cv2.imread(file) for file in files_cat_test])
        test_dog_images_as_np_array = np.asarray([cv2.imread(file) for file in files_dog_test])

        test_cat_images_labels = [{'label': 0, 'src': file} for file in test_cat_images_as_np_array]
        test_dog_images_labels = [{'label': 1, 'src': file} for file in test_dog_images_as_np_array]

        test_data_labels_and_src = test_cat_images_labels + test_dog_images_labels
        np.random.shuffle(test_data_labels_and_src)

        y_test = [item['label'] for item in test_data_labels_and_src]
        x_test = [item['src'] for item in test_data_labels_and_src]

        return [np.asarray(x_train[0:2000]), np.asarray(x_test[0:300]), np.asarray(y_train[0:2000]), np.asarray(y_test[0:300])]














