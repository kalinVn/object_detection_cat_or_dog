from service import NN

nn = NN()

nn.fit()
nn.score()

# visualization
# plot_random_images('cat')
# plot_random_images('dog')

# augmentation_images = Drive.get_images_augmentation('dog')
# plot_augment_images(augmentation_images)