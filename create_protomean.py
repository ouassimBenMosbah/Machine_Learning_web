import subprocess
import os

# Compute the mean image from the training lmdb

class ProtomeanCreater():
    def __init__(self, model):
        originDir = os.getcwd()
        os.chdir(os.environ['CAFFE_ROOT'])
        if (model == "mnist"):
            subprocess.call(["build/tools/compute_image_mean", "examples/mnist/mnist_train_lmdb", "examples/mnist/mean.binaryproto"])
        elif (model == "cifar10"):
            subprocess.call(["build/tools/compute_image_mean", "examples/cifar10/cifar10_train_lmdb", "examples/cifar10/mean.binaryproto"])
        os.chdir(originDir)
