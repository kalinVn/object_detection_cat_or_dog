from service import NN, Drive
from visualizator import plot_random_images, plot_augment_images, plot_image
from config import SRC_FOLDER

# from config import SRC_FOLDER
# import PIL
# from pathlib import Path
#
# path = Path(SRC_FOLDER + 'test/Dog_Resized/').rglob("*.jpg")
# # print(path)
# for img_p in path:
#     try:
#         img = PIL.Image.open(img_p)
#         # print(img)
#     except PIL.UnidentifiedImageError

Drive.resize_images()
nn = NN()

nn.fit()
# nn.score()
input_image_path = 'test_dog_1.jpg'

nn.predict(input_image_path)


# visualization
# img_to_plot = SRC_FOLDER + 'test/Cat/3.jpg'
# plot_image(img_to_plot)
# plot_random_images('cat')
# plot_random_images('dog')

# augmentation_images = Drive.get_images_augmentation('dog')
# plot_augment_images(augmentation_images)

# Drive
# Drive.list_dir_fileNames(SRC_FOLDER + 'test/Cat')
# Drive.resize_images()
# x_train, x_test, y_train, y_test = Drive.train_test_split()

# print(y_test[:6])
