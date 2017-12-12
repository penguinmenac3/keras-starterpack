import os
from scipy.misc import imread
from random import shuffle
import numpy as np


def one_hot(idx, max_idx):
    label = np.zeros(max_idx)
    label[idx] = 1
    return label

def named_folders(base_dir, phase, prepare_features=None):
    classes_dir = os.path.join(base_dir, phase)
    classes = os.listdir(classes_dir)
    images = []
    labels = []
    imgs_per_class = {}
    class_idx = {}
    for c in classes:
        if c not in class_idx:
            class_idx[c] = len(class_idx)
        imgs_per_class[c] = 0
        class_dir = os.path.join(classes_dir, c)
        for filename in os.listdir(class_dir):
            if filename.endswith(".png"):
                feature = imread(os.path.join(class_dir, filename), mode="RGB")
                if prepare_features:
                    feature = prepare_features(feature)
                images.append(feature)
                labels.append(one_hot(class_idx[c], len(classes)))
                imgs_per_class[c] += 1

    return imgs_per_class, np.array(images), np.array(labels), class_idx


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Loading Dataset:")
    imgs_per_class, images, labels, class_idx = named_folders("data/person_classification", "train")

    print("Classes and image count:")
    print(imgs_per_class)

    print("Image shape:")
    print(images[0].shape)

    print(labels[-1])

    for img, class_name in zip(images, labels):
        print(class_name)
        plt.imshow(img)
        plt.show()
