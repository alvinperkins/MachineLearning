import numpy as np
import caffe
import os
os.chdir("/opt/caffe")

from PIL import Image

#
def load_mean(mean_file):
    proto_data = open(mean_file, "rb").read()
    a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    mean  = caffe.io.blobproto_to_array(a)[0]
    return mean

class bvlc:
    def GET(self):

        caffe.set_mode_cpu()


        #load the model
        #net = caffe.Net('models/bvlc_reference_caffenet/deploy.prototxt',
        #                'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
        #                                caffe.TEST)
        net = caffe.Net('models/0179e52305ca768a601f/deploy.prototxt',
                        'models/0179e52305ca768a601f/oxford102.caffemodel',
                                        caffe.TEST)
        #net = caffe.Net('models/c9e99062283c719c03de/deploy_age.prototxt',
        #        'models/c9e99062283c719c03de/age_net.caffemodel',
        #        caffe.TEST)

        # load input and configure preprocessing
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
        #transformer.set_mean('data', load_mean('models/c9e99062283c719c03de/mean.binaryproto').mean(1).mean(1))
        transformer.set_transpose('data', (2,0,1))
        transformer.set_channel_swap('data', (2,1,0))
        transformer.set_raw_scale('data', 255.0)

        #note we can change the batch size on-the-fly
        #since we classify only one image, we change batch size from 10 to 1
        net.blobs['data'].reshape(1,3,227,227)

        #load the image in the data layer
        im = caffe.io.load_image('/tmp/imagesample.jpg')
        net.blobs['data'].data[...] = transformer.preprocess('data', im)

        #compute
        out = net.forward()

        # other possibility : out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))

        #predicted predicted class
        print (out['prob'].argmax())

        #print predicted labels
        labels = np.loadtxt("data/ilsvrc12/synset_words.txt", str, delimiter='\t')
        top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
        print (labels[top_k])
        print str(net.blobs['prob'].data)

class Age():
    def GET(self):
        net_def='models/c9e99062283c719c03de/deploy_age.prototxt'
        net_mod='models/c9e99062283c719c03de/age_net.caffemodel'
        mean_file = 'models/c9e99062283c719c03de/mean.binaryproto'
        net = caffe.Classifier(net_def,net_mod ,
                image_dims=[256,256], mean=load_mean(mean_file).mean(1).mean(1),
                input_scale=None, raw_scale=255.0,
                channel_swap=[2,1,0])
        inputs = [caffe.io.load_image('/tmp/imagesample.jpg')]
        predictions = net.predict(inputs,0)
        print predictions.argmax()

class GoogleNet():

    def __init__(self):
        MODEL_FILE = 'models/bvlc_googlenet/deploy.prototxt'
        PRETRAINED = 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
        caffe.set_mode_cpu()

        self.net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                               mean=np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                               channel_swap=(2,1,0),
                               raw_scale=255,
                               image_dims=(224, 224))

    def predict(self, path):
        input_image = caffe.io.load_image(path)
        #print path
        prediction = self.net.predict([input_image])


        print 'prediction shape:', prediction[0].shape
        print 'predicted class:', prediction[0].argmax(), '\n'
        proba = prediction[0][prediction[0].argmax()]
        ind = prediction[0].argsort()[-5:][::-1] # top-5 predictions
        #print predicted labels
        labels = np.loadtxt("data/ilsvrc12/synset_words.txt", str, delimiter='\t')
        top_k = self.net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
        print (labels[top_k])
        
        return prediction[0].argmax(), proba, ind



o = GoogleNet()
o.predict("/tmp/imagesample.jpg")
