import caffe
import numpy as np

# print "Usage: python convert_protomean.py proto.mean out.npy"

class ProtomeanConverter():

    def __init__(self, protoFile, outputFile):
        blob = caffe.proto.caffe_pb2.BlobProto()
        data = open(protoFile, 'rb').read()
        blob.ParseFromString(data)
        arr = np.array(caffe.io.blobproto_to_array(blob))
        out = arr[0]
        np.save(outputFile, out)
