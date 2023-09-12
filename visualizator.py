from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import random


def plot_augment_images(images):
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    for idx, img in enumerate(images):
        ax[int(idx / 3), idx % 3].imshow(img)
        ax[int(idx / 3), idx % 3].axis('off')
        if idx == 0:
            ax[int(idx / 3), idx % 3].set_title("Original Image")
        else:
            ax[int(idx / 3), idx % 3].set_title("Augmented Image {}".format(idx))

    plt.show()


def plot_random_images(object_class):
    images_path = 'dataset/PetImages/train/Dog/'
    if object_class == 'cat':
        images_path = 'dataset/PetImages/train/Cat/'

    _, _, images = next(os.walk(images_path))
    # prepare 3 x 3 plot total 9 images
    fig, ax = plt.subplots(3, 3, figsize=(20, 10))

    # randomly select and plot an image
    for idx, img in enumerate(random.sample(images, 9)):
        img_read = plt.imread(images_path + img)

        # print(int(idx/3), " ", idx % 3)
        row = int(idx/3)
        col = idx % 3
        ax[row, col].imshow(img_read)
        ax[row, col].axis('off')
        ax[row, col].set_title('Image: ' + img)

    plt.show()

def plot_image(img_path):
    img = mpimg.imread(img_path)
    img_plot = plt.imshow(img)
    plt.show()