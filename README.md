# Keras - Starterpack

This repo aims to contain everything required to quickly develop a deep neural network with keras.
It comes with several dataset loaders and network architectures.
If I can find it the models will also contain pretrained weights.

## Datasets

There are handlers for several datasets.
To get you started quickly.

1. [Custom Classification - Folder per Class](datasets/classification/named_folders.py)
2. [MNIST](datasets/classification/mnist.py)
3. ImageNet [TODO]
4. Coco [TODO]
5. [Cifar10/Cifar100](datasets/classification/cifar.py)
6. [LFW (with named folders)](datasets/classification/named_folders.py)
6. PASCAL VOC [TODO]
7. Places [TODO]
8. Kitti [TODO]
9. Tensorbox [TODO]
10. CamVid [TODO]
11. Cityscapes [TODO]
12. ROS-Robot (as data source) [TODO]

## Models

There are also some models implemented to tinker around with.
Most of the implementations are not done by me from scratch but rather refactoring of online found implementations.
Also the common models will come with pre trained weights I found on the internet.
Just check the comment at the top of their source files.

1. [Alexnet (single stream version)](models/alexnet.py)
2. [VGG 16](models/vgg_16.py)
3. [GoogLeNet (Inception v3)](models/googlenet.py)
4. Overfeat/Tensorbox [TODO]
5. ResNet [TODO]
6. [SegNet](models/segnet.py)
7. Mask RCNN [TODO]
8. monoDepth [TODO]

More non famous models by myself:

1. [CNN for MNIST](models/mnist_cnn.py)
2. [CNN for Person Classification](models/tinypersonnet.py)
3. [CNN for Person Identification [WIP]](models/deeplfw.py)

## Examples

Some samples that should help getting into stuff.

1. [MNIST](examples/mnist.py)
2. [LFW](examples/lfw.py)
3. Imagenet (Baselines) [TODO]
4. Bounding Box Regression [TODO]
5. Segmentations [TODO]
6. Instance Masks [TODO]
7. Reinforcement Learning [TODO]

On non publically availible data:
(however can be used on your own data)

1. [Simple Classification (Folder per Class)](examples/tinypersonnet.py)
