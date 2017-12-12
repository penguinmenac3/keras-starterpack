from models.tinypersonnet import tinypersonnet, prepare_data
from datasets.classification.named_folders import named_folders
from keras.optimizers import SGD
from keras.utils import plot_model


if __name__ == "__main__":
    print("\nLoading Dataset...")
    imgs_per_class, images, labels, class_idx = named_folders("data/person_classification", "train", prepare_data)
    test_imgs_per_class, test_images, test_labels, test_class_idx = named_folders("data/person_classification", "test", prepare_data)

    print("Train Classes and image count:")
    print(imgs_per_class)

    print("Test Classes and image count:")
    print(test_imgs_per_class)

    print("\nCreating Model: tinypersonnet")
    model = tinypersonnet()
    plot_model(model, to_file='models/weights/tinypersonnet.png', show_shapes=True)
    print("Saved structure in: models/weights/tinypersonnet.png")

    print("\nCreate SGD Optimizer")
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    print("\nFit model...")
    model.fit(x=images, y=labels, epochs=10000, validation_data=(test_images, test_labels))
    model.save_weights("models/weights/tinypersonnet.h5")
