import subprocess
import os

class PrototxtConverter():
    def __init__(self, model, solvermode):
        originDir = os.getcwd()
        os.chdir(os.environ['CAFFE_ROOT'])
        if (model == "cifar10quick"):
            files = ["examples/cifar10/cifar10_quick_solver.prototxt", "examples/cifar10/cifar10_quick_solver_lr1.prototxt"]
        elif (model == "cifar10full"):
            files = ["examples/cifar10/cifar10_full_solver.prototxt", "examples/cifar10/cifar10_full_solver_lr2.prototxt", "examples/cifar10/cifar10_full_solver_lr1.prototxt"]
        else:
            files = ["examples/mnist/lenet_solver.prototxt"]
        for file in files:
            if (solvermode == "cpu"):
                subprocess.call(["sed", "-i", "s/solver_mode:\ GPU/solver_mode:\ CPU/g", file])
            else:
                subprocess.call(["sed", "-i", "s/solver_mode:\ CPU/solver_mode:\ GPU/g", file])
            subprocess.call(["sed", "-i", "s/snapshot_format:\ HDF5/snapshot_format:\ BINARYPROTO/g", file])
        os.chdir(originDir)
