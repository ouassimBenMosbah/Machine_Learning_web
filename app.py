#!/usr/bin/python
# -*- coding: utf-8 -*-
import caffe
import classify  # Use classify.py file as a module
import create_protomean  # Use create_protomean.py file as a module
import convert_protomean  # Use convert_protomean.py file as a module
import convert_prototxt  # Use convert_prototxt.py file as a module
import cPickle
import cStringIO as StringIO
import datetime
import exifutil
import fileinput
import flask
from flask import request
import logging
import numpy as np
import os
import optparse
import pandas as pd
from PIL import Image
import psutil
import subprocess
from tensorflow.python.client import device_lib
import threading
import time
import tornado.wsgi
import tornado.httpserver
import urllib
import werkzeug
import wget

REPO_DIRNAME = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + '/../..')
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

# Obtain the flask app object
# Flask will be used for routing + templates
app = flask.Flask(__name__)


# index view
@app.route('/index')
@app.route('/home')
@app.route('/')
def index():
    """
    Show the index template.
    """
    return flask.render_template('index.html', has_result=False)


# dataset view
@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    """
    Show the dataset template.
    """
    tmp = False
    download_progression = []
    try:
        download_progression = get_download_progression()
        tmp = True
    except IOError:
        pass
    return flask.render_template('dataset.html', infos=['GET', tmp, False,
                                                        download_progression])


# create_standard_dataset view
@app.route('/create_standard_dataset', methods=['GET', 'POST'])
def create_standard_dataset():
    """
    Download and store the datasets wanted in the LMDB database if it wasn't
    already done and it shows the dataset template.
    """
    threads = []
    scripts = []
    method = flask.request.method
    tmp = False
    download_progression = []

    if method == 'POST':
        try:
            download_progression = get_download_progression()
            tmp = True
        except IOError:
            pass
        return flask.render_template('dataset.html',
                                     infos=[method, tmp, (len(scripts) == 0),
                                            download_progression])

    elif method == 'GET':
        try:
            download_progression = get_download_progression()
            tmp = True
        except IOError:
            cifar10 = flask.request.args.get("cifar10",
                                             default=None, type=None)
            mnist = flask.request.args.get("mnist", default=None, type=None)
            if (mnist == "true"):
                if not (os.path.exists((os.environ['CAFFE_ROOT']) +
                        '/examples/mnist/mnist_test_lmdb') and
                        os.path.exists((os.environ['CAFFE_ROOT']) +
                        '/examples/mnist/mnist_train_lmdb')):
                    scripts.append(['./data/mnist/get_mnist.sh',
                                    './examples/mnist/create_mnist.sh'])
                if not os.path.exists((os.environ['CAFFE_ROOT']) +
                                      '/examples/mnist/mean.binaryproto'):
                    create_protomean.ProtomeanCreater("mnist")
                if not os.path.exists((os.environ['CAFFE_ROOT']) +
                                      '/examples/mnist/mean.npy'):
                    convert_protomean.ProtomeanConverter(
                        (os.environ['CAFFE_ROOT']) +
                        '/examples/mnist/mean.binaryproto',
                        (os.environ['CAFFE_ROOT'])+'/examples/mnist/mean.npy')
            if (cifar10 == "true"):
                if not (os.path.exists((os.environ['CAFFE_ROOT']) +
                        '/examples/cifar10/cifar10_test_lmdb') and
                        os.path.exists((os.environ['CAFFE_ROOT']) +
                        '/examples/cifar10/cifar10_train_lmdb') and
                        os.path.exists((os.environ['CAFFE_ROOT']) +
                        '/examples/cifar10/mean.binaryproto')):
                    scripts.append(['./data/cifar10/get_cifar10.sh',
                                    './examples/cifar10/create_cifar10.sh'])
                if not os.path.exists((os.environ['CAFFE_ROOT']) +
                                      '/examples/cifar10/mean.binaryproto'):
                    create_protomean.ProtomeanCreater("cifar10")
                if not os.path.exists((os.environ['CAFFE_ROOT']) +
                                      '/examples/cifar10/mean.npy'):
                    convert_protomean.ProtomeanConverter(
                        (os.environ['CAFFE_ROOT']) +
                        '/examples/cifar10/mean.binaryproto',
                        (os.environ['CAFFE_ROOT']) +
                        '/examples/cifar10/mean.npy')
            t = threading.Thread(target=dataset_worker, args=(scripts,))
            threads.append(t)
            t.start()
        return flask.render_template('dataset.html', infos=[method, tmp,
                                     (len(scripts) == 0),
                                     download_progression])


def dataset_worker(scripts):
    """
    The handler that launch scripts to download and store datasets.
    """
    originDir = os.getcwd()
    os.chdir(os.environ['CAFFE_ROOT'])
    for g, c in scripts:
        subprocess.call([g])
        subprocess.call([c])
    os.chdir(originDir)
    return


def get_download_progression():
    """
    Getter that return the informations about the dataset download progression.
    """
    with open((os.environ['CAFFE_ROOT'])+"/tmp.txt", 'rb') as fh:
        fh.seek(-1024, os.SEEK_END)
        download_progression = fh.readlines()[-2].decode().replace('.', '')
        return map(str, ' '.join(download_progression.split()).split(' '))


# training_page view
@app.route('/training_page')
def training_page():
    """
    Show the training page template.
    """
    existingDataset = get_existing_dataset()
    stats = gpu_cpu_stats()
    stats.append(time.strftime("%c"))
    return flask.render_template('training_page.html', stats=stats,
                                 existingDataset=existingDataset,
                                 information="", information_title="")


def get_existing_dataset():
    """
    Return the existing datasets in the CAFFE_ROOT directory.
    """
    existingDataset = {}
    if (os.path.exists((os.environ['CAFFE_ROOT']) +
                       '/examples/mnist/mnist_test_lmdb') and
        os.path.exists((os.environ['CAFFE_ROOT']) +
                       '/examples/mnist/mnist_train_lmdb')):
        if 'mnist' not in existingDataset:
            existingDataset['mnist'] = \
                (os.environ['CAFFE_ROOT'])+'/examples/mnist/'
    if (os.path.exists((os.environ['CAFFE_ROOT']) +
                       '/examples/cifar10/cifar10_test_lmdb') and
        os.path.exists((os.environ['CAFFE_ROOT']) +
                       '/examples/cifar10/cifar10_train_lmdb') and
        os.path.exists((os.environ['CAFFE_ROOT']) +
                       '/examples/cifar10/mean.binaryproto')):
        if 'cifar10' not in existingDataset:
            existingDataset['cifar10'] = \
                (os.environ['CAFFE_ROOT'])+'/examples/cifar10/'
    return existingDataset


@app.route('/get_train_stats', methods=['POST'])
def train_stats():
    """Show the training page template with informations about CPU and GPU
    statistics and about training statistics if a training was launched."""
    stats = gpu_cpu_stats()
    stats.append(time.strftime("%c"))
    information_title = 'Training '
    information = 'Training initialization'
    try:
        for line in open(os.environ['CAFFE_ROOT']+"/tmp2.txt").readlines():
            if "snapshot_prefix" in line:
                information_title += line.split('/')[1]
            elif "solver_mode" in line:
                information_title = \
                    information_title + " with " + line.split()[1]
                break
        for line in reversed(open(
                            os.environ['CAFFE_ROOT']+"/tmp2.txt").readlines()):
            if ':218' in line:
                information = line[line.find('Iteration'):].\
                    replace('s,', 's |').replace('(', '| ').\
                    replace('),', ' |').replace('I', 'Number of i', 1)
                break
    except IOError:
        information = 'The server is not training'
    return flask.render_template('training_page.html', stats=stats,
                                 information=information,
                                 information_title=information_title)


# create_standard_database view
@app.route('/train_standard_model', methods=['POST'])
def train_standard_model():
    """
    Train the application with CPU or GPU for a given dataset if there isn't
    another dataset training. Then it shows the training page template.
    """
    threads = []
    model = request.form['model']
    solvermode = request.form['solvermode']
    if (solvermode == "gpu"):
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    if not os.path.exists((os.environ['CAFFE_ROOT'])+"/tmp2.txt"):
        if (model == "cifar10quick"):
            convert_prototxt.PrototxtConverter(model, solvermode)
            script = 'examples/cifar10/train_quick.sh'
        elif (model == "cifar10full"):
            convert_prototxt.PrototxtConverter(model, solvermode)
            script = 'examples/cifar10/train_full.sh'
        elif (model == "mnist"):
            convert_prototxt.PrototxtConverter(model, solvermode)
            script = 'examples/mnist/train_lenet.sh'
        t = threading.Thread(target=train_worker, args=(script,))
        threads.append(t)
        t.start()
    existing_dataset = get_existing_dataset()
    stats = gpu_cpu_stats()
    stats.append(time.strftime("%c"))
    return flask.render_template('training_page.html', stats=stats,
                                 existingDataset=existing_dataset,
                                 information="", information_title="")


def train_worker(script):
    """
    The handler that launch scripts to train the server.
    """
    originDir = os.getcwd()
    os.chdir(os.environ['CAFFE_ROOT'])
    stderr = open("tmp2.txt", "wb+")
    subprocess.call(script, shell=True, stderr=stderr)
    stderr.close()
    os.remove("tmp2.txt")
    os.chdir(originDir)
    return


# image_detection view
@app.route('/image_detection')
def image_detection():
    """
    Shows the image detection template.
    """
    existingModel = {}
    if os.path.exists((os.environ['CAFFE_ROOT']) +
                      '/examples/mnist/lenet_iter_10000.caffemodel'):
        if 'mnist' not in existingModel:
            existingModel['mnist'] = \
                (os.environ['CAFFE_ROOT'])+'/examples/mnist/'
    if os.path.exists((os.environ['CAFFE_ROOT']) +
                      '/examples/cifar10/cifar10_quick_iter_5000.caffemodel'):
        if 'cifar10quick' not in existingModel:
            existingModel['cifar10quick'] = \
                (os.environ['CAFFE_ROOT'])+"/examples/cifar10/"
    if os.path.exists(
            (os.environ['CAFFE_ROOT']) +
            '/examples/cifar10/cifar10_full_iter_70000.caffemodel'):
        if 'cifar10full' not in existingModel:
            existingModel['cifar10full'] = \
                (os.environ['CAFFE_ROOT'])+"/examples/cifar10/"
    return flask.render_template('image_detection.html', has_result=False,
                                 existingModel=existingModel)


# classify_url view
@app.route('/classify_url', methods=['POST'])
def classify_url():
    """
    Predict the content of an image given from an URL
    and show the image detection template.
    """
    imageurl = request.form['imageurl']
    model = request.form['model']
    solvermode = request.form['solvermode']
    try:
        filename = wget.download(imageurl, out=UPLOAD_FOLDER)
    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image download error: %s', err)
        return flask.render_template(
            'image_detection.html', has_result=True,
            result=(False, 'Cannot download image from URL.')
        )
    if (model == "cifar10quick"):
        convert_prototxt.PrototxtConverter(model, solvermode)
        starttime = time.time()
        classifier = classify.Classifier(solvermode)
        result = \
            classifier.classify_image(
                (os.environ['CAFFE_ROOT']) +
                '/examples/cifar10/cifar10_quick.prototxt',
                (os.environ['CAFFE_ROOT']) +
                '/examples/cifar10/cifar10_quick_iter_5000.caffemodel',
                (os.environ['CAFFE_ROOT'])+'/examples/cifar10/mean.npy',
                filename,
                (os.environ['CAFFE_ROOT']) +
                '/data/cifar10/batches.meta.txt', False)
        endtime = time.time()
        totaltime = round(endtime - starttime, 4)
    if (model == "cifar10full"):
        convert_prototxt.PrototxtConverter(model, solvermode)
        starttime = time.time()
        classifier = classify.Classifier(solvermode)
        result = \
            classifier.classify_image(
                (os.environ['CAFFE_ROOT']) +
                '/examples/cifar10/cifar10_full.prototxt',
                (os.environ['CAFFE_ROOT']) +
                '/examples/cifar10/cifar10_full_iter_70000.caffemodel',
                (os.environ['CAFFE_ROOT'])+'/examples/cifar10/mean.npy',
                filename, (os.environ['CAFFE_ROOT']) +
                '/data/cifar10/batches.meta.txt', False)
        endtime = time.time()
        totaltime = round(endtime - starttime, 4)
    if (model == "mnist"):
        convert_prototxt.PrototxtConverter(model, solvermode)
        starttime = time.time()
        classifier = classify.Classifier(solvermode)
        result = \
            classifier.classify_image(
                (os.environ['CAFFE_ROOT']) +
                '/examples/mnist/lenet.prototxt',
                (os.environ['CAFFE_ROOT']) +
                '/examples/mnist/lenet_iter_10000.caffemodel', None, filename,
                None, True)
        endtime = time.time()
        totaltime = round(endtime - starttime, 4)
    return flask.render_template('image_detection.html', has_result=True,
                                 result=result, imagesrc=imageurl,
                                 totaltime=totaltime)


# classify_upload view
@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    """
    Predict the content of an image given from the local computer
    and show the image detection template.
    """
    model = request.form['model']
    solvermode = request.form['solvermode']
    try:
        # We will save the file to disk for possible data collection.
        imagefile = request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image = exifutil.open_oriented_im(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'image_detection.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    if (model == "cifar10quick"):
        convert_prototxt.PrototxtConverter(model, solvermode)
        starttime = time.time()
        classifier = classify.Classifier(solvermode)
        result = \
            classifier.classify_image(
                (os.environ['CAFFE_ROOT']) +
                '/examples/cifar10/cifar10_quick.prototxt',
                (os.environ['CAFFE_ROOT']) +
                '/examples/cifar10/cifar10_quick_iter_5000.caffemodel',
                (os.environ['CAFFE_ROOT'])+"/examples/cifar10/mean.npy",
                filename,
                (os.environ['CAFFE_ROOT'])+"/data/cifar10/batches.meta.txt",
                False)
        endtime = time.time()
        totaltime = round(endtime - starttime, 4)
    if (model == "cifar10full"):
        convert_prototxt.PrototxtConverter(model, solvermode)
        starttime = time.time()
        classifier = classify.Classifier(solvermode)
        result = \
            classifier.classify_image(
                (os.environ['CAFFE_ROOT']) +
                "/examples/cifar10/cifar10_full.prototxt",
                (os.environ['CAFFE_ROOT']) +
                "/examples/cifar10/cifar10_full_iter_70000.caffemodel",
                (os.environ['CAFFE_ROOT'])+"/examples/cifar10/mean.npy",
                filename,
                (os.environ['CAFFE_ROOT'])+"/data/cifar10/batches.meta.txt",
                False)
        endtime = time.time()
        totaltime = round(endtime - starttime, 4)
    if (model == "mnist"):
        convert_prototxt.PrototxtConverter(model, solvermode)
        starttime = time.time()
        classifier = classify.Classifier(solvermode)
        result = \
            classifier.classify_image(
                (os.environ['CAFFE_ROOT'])+'/examples/mnist/lenet.prototxt',
                (os.environ['CAFFE_ROOT']) +
                '/examples/mnist/lenet_iter_10000.caffemodel',
                None, filename, None, True)
        endtime = time.time()
        totaltime = round(endtime - starttime, 4)
    return flask.render_template('image_detection.html', has_result=True,
                                 result=result,
                                 imagesrc=embed_image_html(image),
                                 totaltime=totaltime)


# about view
@app.route('/about')
def about():
    """
    Show the about template.
    """
    return flask.render_template('about.html')


# contact view
@app.route('/contact')
def contact():
    """
    Show the contact template.
    """
    return flask.render_template('contact.html')


# customized_errors view
@app.errorhandler(401)
@app.errorhandler(404)
@app.errorhandler(500)
def customized_errors(error):
    """
    Show the error template.
    """
    return flask.render_template('error.html', num_error=error.code)


def embed_image_html(image):
    """
    Creates an image embedded in HTML base64 format
    """
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    """
    Check if the file given for the detection is allowed
    """
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


def start_tornado(app, port=5000):
    """
    Start tornado as WSGI server
    """
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()

    # Initialize classifier + warm start by forward for allocation
    """app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    app.clf.net.forward()
    """
    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


def get_available_gpus():
    """
    Return a list of boolean to tell you either if there are GPUs or not.
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def gpu_cpu_stats():
    """
    Get GPUs and CPUs stats.
    """
    resultat = [[], []]
    boolGPU = False
    boolCPU = False

    if get_available_gpus():
            # If GPU available, write it datas
            p = subprocess.Popen(['gpustat', '--no-color', '-c'],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            output, err = \
                p.communicate("input data that is passed to subprocess' stdin")
            resultat[0] = [output.replace("\n", "<br>")]

    # psutil.cpu_count(logical=True:
    # return the number of logical CPUs available for this program
    # tot_m    Total installed memory (MemTotal and SwapTotal in /proc/meminfo)
    # used_m   Used memory (calculated as total - free - buffers - cache)
    # free_m   Unused memory (MemFree and SwapFree in /proc/meminfo)
    tot_m, used_m, free_m = os.popen('free -tmh').read().split()[-3:]
    pe = str(psutil.virtual_memory()).index("percent")
    str_percent_use = str(psutil.virtual_memory())[pe+8:pe+12]
    percent_use = int(str_percent_use[:str_percent_use.index('.')])
    resultat[1] = [psutil.cpu_count(logical=True), tot_m, used_m, percent_use]

    if len(resultat[0]) > 0:
        boolGPU = True
    if len(resultat[1]) > 0:
        boolCPU = True
    return [boolGPU, resultat[0], boolCPU, resultat[1]]


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
start_from_terminal(app)
