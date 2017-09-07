#!/usr/bin/env sh
# This scripts downloads the CIFAR10 (binary version) data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $CAFFE_ROOT/data
if [ ! -d "cifar100" ]; then
  mkdir cifar100
fi

echo "Downloading..."

wget --no-check-certificate http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz -o "$CAFFE_ROOT/tmp.txt"

echo "Unzipping..."

tar -xf cifar-100-binary.tar.gz && rm -f cifar-100-binary.tar.gz
mv cifar-100-batches-bin/* . && rm -rf cifar-100-batches-bin

# Creation is split out because leveldb sometimes causes segfault
# and needs to be re-created.

echo "Done."
rm -f "$CAFFE_ROOT/tmp.txt"
