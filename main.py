import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

image_paths = [f"in_images/{i+1}.png" for i in range(6)]
images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
binary_images = [cv2.bitwise_not(cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]) for image in images]

objects_coordinates = []
for binary_image in binary_images:
    object_matrix = (binary_image / 255).astype(np.uint8)
    coordinates = np.argwhere(object_matrix == 1)
    objects_coordinates.append(coordinates)

pca_components = []
for coordinates in objects_coordinates:
    pca = PCA(n_components=2)
    pca.fit(coordinates)
    pca_components.append(pca.components_)

for image, coordinates, components in zip(images, objects_coordinates, pca_components):
    plt.figure()
    plt.imshow(image, cmap='gray')

    origin = np.mean(coordinates, axis=0)
    plt.quiver(*origin[::-1], components[0, 0], components[0, 1], color='r', scale=3)
    plt.quiver(*origin[::-1], components[1, 0], components[1, 1], color='b', scale=3)

    plt.gca().invert_yaxis()
    plt.show()
