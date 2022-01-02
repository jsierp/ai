# GAN on MNIST
Simple implementation of GAN on MNIST dataset.
What I've learned:
* It takes some time at the beginning for discriminator to train. Only after that the generator begins to train.
* Make sure your dataset is properly normalized/preprocessed. It took me a while to figure out that MNIST images pixel values are already in [0, 1].

Todo: add a script version, describe the algorithm.
