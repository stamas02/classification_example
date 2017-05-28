import keras.backend as K
from pkg_resources import parse_version
import numpy as np
import tensorflow as tf
class Log:
    def __init__(self, log_dir):
        self.viz = None
        if K._BACKEND == 'tensorflow':
            self.viz = TensorBoardLog(log_dir)
        else:
            raise "you need to set tensorboard as a backend in order to use tensorboard"

    def log(self, model,  images, scalars, epoch):
        self.viz.log(model,  images, scalars, epoch)

    def set_model(self, model):
        self.viz.set_model(model)


class TensorBoardLog:
    def __init__(self,log_dir):
        self.merged = None
        self.write_images = False
        self.write_graph = True
        self.log_dir = log_dir

    def set_model(self, model):
        import tensorflow as tf
        import keras.backend.tensorflow_backend as KTF

        self.model = model
        self.sess = KTF.get_session()
        if self.merged is None:
            for layer in self.model.layers:

                for weight in layer.weights:
                    tf.summary.histogram(weight.name, weight)

                    if self.write_images:
                        w_img = tf.squeeze(weight)

                        shape = w_img.get_shape()
                        if len(shape) > 1 and shape[0] > shape[1]:
                            w_img = tf.transpose(w_img)

                        if len(shape) == 1:
                            w_img = tf.expand_dims(w_img, 0)

                        w_img = tf.expand_dims(tf.expand_dims(w_img, 0), -1)

                        tf.summary.image(weight.name, w_img)

                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name),
                                         layer.output)

        self.merged = tf.summary.merge_all()

        if self.write_graph:
            if parse_version(tf.__version__) >= parse_version('0.8.0'):
                self.writer = tf.summary.FileWriter(self.log_dir,
                                                     self.sess.graph)
            else:
                self.writer = tf.summary.FileWriter(self.log_dir,
                                                     self.sess.graph_def)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

    def log(self, model,  images, scalars, epoch):
        import tensorflow as tf
        s = np.array(self.model.input._shape.dims)
        s[0] = 1
        feed_dict = dict(zip( self.model.inputs, (np.zeros(s,dtype=np.float32),np.zeros((s[0]), dtype=np.float32))))
        feed_dict[K.learning_phase()] = 0
        result = self.sess.run([self.merged], feed_dict=feed_dict)
        summary_str = result[0]
        self.writer.add_summary(summary_str, epoch)


        for key, value in images.items():
            value = np.array([value])
            tf_image = tf.placeholder(np.uint8, value.shape,name="input")
            summary_op = tf.summary.image(key+str(epoch)   , tf_image)
            summary = self.sess.run(summary_op, feed_dict={tf_image: value})
            self.writer.add_summary(summary, epoch)

        summary = tf.Summary()
        for key, value in scalars.items():
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = key
        self.writer.add_summary(summary, epoch)


        #self.writer.flush()
        #self.writer.close()

