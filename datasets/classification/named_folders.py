import os
from scipy.misc import imread
from random import shuffle


def named_folders(base_dir, phase):
    classes_dir = os.path.join(base_dir, phase)
    classes = os.listdir(classes_dir)
    images = []
    imgs_per_class = {}
    for c in classes:
        imgs_per_class[c] = 0
        class_dir = os.path.join(classes_dir, c)
        for filename in os.listdir(class_dir):
            if filename.endswith(".png"):
                images.append((os.path.join(class_dir, filename), c))
                imgs_per_class[c] += 1

    def gen():
        while True:
            # Shuffle data
            shuffle(images)
            for date in images:
                yield (imread(date[0], mode="RGB"), date[1])

    return imgs_per_class, gen()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    imgs_per_class, gen = named_folders("data/person_classification", "train")

    print("Classes and image count:")
    print(imgs_per_class)

    img, class_name = next(gen)
    print("Image shape:")
    print(img.shape)

    for img, class_name in gen:
        plt.imshow(img)
        plt.show()
