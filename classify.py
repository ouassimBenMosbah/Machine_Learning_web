# set up Python environment: numpy for numerical routines
import numpy as np
# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
import caffe
import os
import cv2
#if len(sys.argv) != 7:
#            print "Usage: python classify.py cpu/gpu *.prototxt *.caffemodel *.npy *.jpg *.txt"
#           sys.exit()

class Classifier():

    def __init__(self, cpuMode):
        if cpuMode == 'cpu':
            caffe.set_mode_cpu()
        if cpuMode == 'gpu':
            caffe.set_mode_gpu()

    def classify_image(self, model_def, model_weights, mean_file, image_file, labels_file, greyscale):

        net = caffe.Net(model_def,   # defines the structure of the model
                        1,
                        weights=model_weights)  # contains the trained weights

        if greyscale:
            print image_file
            img = cv2.imread(image_file, 0)
            if img.shape != [28, 28]:
                img2 = cv2.resize(img, (28, 28))
                img = img2.reshape(28, 28, -1)
            else:
                img = img.reshape(28, 28, -1)
            #revert the image,and normalize it to 0-1 range
            img = 1.0 - img/255.0
            out = net.forward_all(data=np.asarray([img.transpose(2, 0, 1)]))

            res = []
            res.append([out['prob'][0].argmax(), 99])
        else:
            # load the mean image for subtraction
            mu = np.load(mean_file)
            mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
            print 'mean-subtracted values:', zip('BGR', mu)

            # create transformer for the input called 'data'
            transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

            transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
            transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
            transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
            transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

            # set the size of the input (we can skip this if we're happy
            #  with the default; we can also change it later, e.g., for different batch sizes)
            net.blobs['data'].reshape(1, 3, 32, 32)
            image = caffe.io.load_image(image_file)
            transformed_image = transformer.preprocess('data', image)

            # copy the image data into the memory allocated for the net
            net.blobs['data'].data[...] = transformed_image

            ## perform classification
            output = net.forward()

            output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

            print 'predicted class is:', output_prob.argmax()

            res = []
            # load labels

            labels = np.loadtxt(labels_file, str, delimiter='\t')

            print 'output label:', labels[output_prob.argmax()]

            # sort top five predictions from softmax output
            top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items
            label_top_inds = labels[top_inds]

            for i in range(5):
                res.append([label_top_inds.item(i), round(output_prob[top_inds].item(i)*100, 3)])

        return res
