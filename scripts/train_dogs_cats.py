import tensorflow as tf 
import numpy as np 
import os 
from imutils import paths
import argparse
from sklearn.preprocessing import LabelEncoder
from random import shuffle
import cv2
import time
import matplotlib.pyplot as plt
from datetime import timedelta
import datetime
import pylab

## Paths

log_path = "/tmp/tensorflow/log/"
path_to_train_images = "/home/ec2-user/data/train/"
path_to_test_images = "/home/ec2-user/data/test"
path_to_models = "/home/ec2-user/models/"

## TODO Convert network configurations into a dict. Integrate batch normalization into the network

class Network:

    def __init__(self, path_dict):
        # Hyperparameters
        self.num_channels = 3
        self.img_size = 128
        self.filter_size = 3
        self.num_filters1 = 16
        self.num_filters2 = 32
        self.num_filters3 = 32
        self.num_filters4 = 16
        self.fc_size1 = 1024
        self.fc_size2 = 512
        self.dropout_probability = 0.6
        self.num_classes = 2
        self.learning_rate = 0.01
        self.regularizer_rate = 5e-2
        self.batch_size = 128


        imagePaths = list(paths.list_images(path_dict['train']))
        self.testImagePaths = list(paths.list_images(path_dict['test']))
        shuffle(imagePaths)
        shuffle(self.testImagePaths)
        print("# of images: {0}".format(len(imagePaths)))
        print("# of test images: %d " % len(self.testImagePaths))

        valid_size = int(0.05 * len(imagePaths))
        trainig_size = len(imagePaths) - valid_size
        print("Training size: {0}, Validation size: {1}".format(trainig_size, valid_size))
        self.valImagePaths = imagePaths[trainig_size:]
        self.trainImagePaths = imagePaths[:trainig_size]

        self.path_to_models = path_dict['models']
        self.num_epochs = 5
        self.regularizer = None 
        self.use_batchnorm = False
        self.parse_arguments()

        print("Network Info: \nLearning Rate: {0:2.5f}\nUse BatchNorm: {1}\n # of epochs: {2}\n"
            "Batch Size: {3}".format(self.learning_rate, self.use_batchnorm, self.num_epochs, self.batch_size))

    def parse_arguments(self):
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-lr", "--learningrate", required=False, type=float)
        self.ap.add_argument("-ur", "--usereg", required=False, type=int, help="Flag to use regularizer")
        self.ap.add_argument("-ubn", "--usebatchnorm", required=False, type=int, help="Flag to use batch norm")
        self.ap.add_argument("-epnum", "--numepochs", required=False, type=int, help="Number of epochs to train")
        self.ap.add_argument("-istr", "--istraining", required=False, type=int)
        args = vars(self.ap.parse_args())

        if args["learningrate"] != None:
            self.learning_rate = args["learningrate"]
        if args["usereg"]:
            self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.regularizer_rate)
            print("Using L2 Regularization with rate {0:.4}".format(self.regularizer_rate))
        if args["usebatchnorm"] != None:
            bn_in = args["usebatchnorm"]
            if bn_in == 1:
                self.use_batchnorm = True
                print("Using Batch Norm")
            else:
                self.use_batchnorm = False
                print("Not using Batch Norm")
        if args["numepochs"] != None:
            self.num_epochs = args["numepochs"]

    def init_weights(self, shape, init_method, scope):
        with tf.variable_scope(scope):
            if init_method == "zeros":
                print("Giving zeros to ", scope)
                return tf.Variable(tf.zeros(shape=shape, dtype=tf.float32), name="biases")
            if init_method == "normal":
                print("Giving normal to ", scope)
                return tf.get_variable(name="weights", shape=shape, dtype=tf.float32, 
                                        initializer=tf.random_normal_initializer(), regularizer=self.regularizer)
            if init_method == "xavier":
                print("Giving xavier to ", scope)
                (fan_in, fan_out) = Network.get_fans(shape)
                interval = np.sqrt(6.0/(fan_in+fan_out))
                return tf.get_variable(name="weights", shape=shape, dtype=tf.float32,
                                        initializer=tf.random_uniform_initializer(minval=-interval, maxval=interval),
                                            regularizer=self.regularizer)
            if init_method == "heinit":
                print("Giving heinit to ", scope)
                (fan_in, fan_out) = Network.get_fans(shape)
                interval = np.sqrt(12.0/(fan_in+fan_out))
                return tf.get_variable(name="weights", shape=shape, dtype=tf.float32,
                                        initializer=tf.random_uniform_initializer(minval=-interval, maxval=interval),
                                            regularizer=self.regularizer)

    # Config dict contains num_input_channels, filter_size, num_filters, scope, init_method, useBN
    def new_conv_layer(self, input, config, use_pooling=True):
        shape = [config["filter_size"], config["filter_size"], config["num_input_channels"], config["num_filters"]]
        weights = self.init_weights(shape, init_method=config["init_method"], scope=config["scope"])
        biases = self.init_weights(shape=[config["num_filters"]], init_method="zeros", scope=config["scope"])
        with tf.variable_scope(config["scope"]):
            layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
            layer += biases 
            if self.use_batchnorm is True:
                print("Applying BN to ", config["scope"])
                layer = self.batch_norm_wrapper(inputs=layer, scope=config["scope"], 
                                            is_training=config["is_training"])

            layer = tf.nn.relu(layer)
            if use_pooling:
                layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        return layer, weights

        # Config dict contains num_inputs, num_outputs, init_method, scope, useBN
    def new_fc_layer(self, input, config, use_relu=True):
        weights = self.init_weights(shape=(config["num_inputs"], config["num_outputs"]), 
                                init_method=config["init_method"], scope=config["scope"])
        biases = self.init_weights(shape=[config["num_outputs"]], init_method="zeros", scope=config["scope"])
        with tf.variable_scope(config["scope"]):
            layer = tf.matmul(input, weights) + biases
            if self.use_batchnorm is True:
                print("Applying BN to ", config["scope"])
                layer = self.batch_norm_wrapper(inputs=layer, scope=config["scope"], 
                                    is_training=config["is_training"])
            if use_relu:
                layer = tf.nn.relu(layer)
        return layer, weights

    # contd: continue from last trained point, restore parameters from path_to_models
    def train_network(self, contd=False):
        (self.x_, self.y_), self.train_step, self.accuracy, _, self.cost, self.dcl, self.cost_r, self.dropout_prob = self.build_graph(is_training=True)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        if contd == True:
            print("Restoring paramaters before training..")
            saver = tf.train.Saver()
            saver.restore(self.sess, self.path_to_models)
            # graph = tf.get_default_graph()
            # test_pop_mean = graph.get_tensor_by_name("ConvLayer2_2/ConvLayer2/pop_mean:0")
        print("Variables:")
        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(i.name)

        self.optimize()

        graph = tf.get_default_graph()

        saver = tf.train.Saver()
        saver.save(self.sess, self.path_to_models)

        self.sess.close()
        tf.reset_default_graph()

    def build_graph(self, is_training):
        # Placeholders
        x_ = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.num_channels], name='x-input')
        y_ = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y-labels')
        y_true_class_encoder = tf.argmax(y_, axis=1)
        keep_prob = tf.placeholder(tf.float32, name="keep_prob_dropout")
        self.is_training_pc = tf.placeholder(tf.bool, name="is_training_pc")
         # Layers
        layer_conv1, weights = self.new_conv_layer(input=x_, config={"num_input_channels":self.num_channels, 
                                                "filter_size":self.filter_size, "num_filters":self.num_filters1, 
                                                "scope":"ConvLayer1", "init_method":"heinit", 
                                                "is_training":is_training}, use_pooling=False)

        print("Constructed ", layer_conv1)
        
        layer_conv2, weights = self.new_conv_layer(input=layer_conv1, config={"num_input_channels":self.num_filters1, 
                                                "filter_size":self.filter_size, "num_filters":self.num_filters2, 
                                                "scope":"ConvLayer2", "init_method":"heinit",
                                                "is_training":is_training}, use_pooling=True)
        #with tf.variable_scope("Conv2_Dropout"):
            #layer_conv2 = tf.nn.dropout(layer_conv2, keep_prob)

        print("Constructed ", layer_conv2)
        
        layer_conv3, weights = self.new_conv_layer(input=layer_conv2, config={"num_input_channels":self.num_filters2, 
                                                "filter_size":self.filter_size,"num_filters":self.num_filters3, 
                                                "scope":"ConvLayer3", "init_method":"heinit", 
                                                "is_training":is_training},  use_pooling=True)
        with tf.variable_scope("Conv3_Dropout"):
            layer_conv3 = tf.nn.dropout(layer_conv3, keep_prob)
        print("Constructed ", layer_conv3)
        
        layer_conv4, weights = self.new_conv_layer(input=layer_conv3, config={"num_input_channels":self.num_filters3, 
                                              "filter_size":self.filter_size,"num_filters":self.num_filters4, 
                                              "scope":"ConvLayer4", "init_method":"normal", 
                                              "is_training":is_training}, use_pooling=True)
        #with tf.variable_scope("Conv4_Droppout"):
         #   layer_conv4 = tf.nn.dropout(layer_conv4, keep_prob)
        print("Constructed ", layer_conv4)

        layer_flat, self.num_features = Network.flatten_layer(layer_conv4, scope="Flatten")
        # layer_flat, num_features = flatten_layer(x_)
        print("Number of features before FCs ", self.num_features)

        layer_fc1, weights = self.new_fc_layer(input=layer_flat, config={"num_inputs":self.num_features, "num_outputs":self.fc_size1,
                                                "scope":"FCLayer1", "init_method":"heinit", 
                                                "is_training":is_training},  use_relu=True)

        with tf.variable_scope("FC1_Dropout"):
            layer_fc1 = tf.nn.dropout(layer_fc1, keep_prob)
        print("Constructed ", layer_fc1)

        layer_fc2, weights = self.new_fc_layer(input=layer_fc1, config={"num_inputs":self.fc_size1, "num_outputs":self.fc_size2,
                                                "scope":"FCLayer2", "init_method":"heinit",
                                                 "is_training":is_training}, use_relu=True)

        with tf.variable_scope("FC2_Dropout"):
            layer_fc2 = tf.nn.dropout(layer_fc2, keep_prob)

        layer_fc3, weights = self.new_fc_layer(input=layer_fc2, config={"num_inputs":self.fc_size2, "num_outputs":self.num_classes,
                                                       "scope":"FCLayer3", "init_method":"heinit",
                                                        "is_training":is_training}, use_relu=False)
        # with tf.variable_scope("FC3_Dropout"):
        #     layer_fc3 = tf.nn.dropout(layer_fc3, keep_prob)


        # Predictions
        y_pred = tf.nn.softmax(layer_fc3, name="y-predictions")
        y_pred_class = tf.argmax(y_pred, axis=1, name="y-predicted-classes")

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3, labels=y_, name="cross_entropy")
        cost = tf.reduce_mean(cross_entropy, name="cost")
        reg_losses =  tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        cost_reg = cost + self.regularizer_rate * np.sum(reg_losses)

        global_step = tf.Variable(0., trainable=False)
        decay_learning_rate = tf.train.exponential_decay(self.learning_rate, global_step,
                                                    125, 0.96, staircase=False)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.RMSPropOptimizer(learning_rate=decay_learning_rate).minimize(cost_reg, global_step=global_step)

        correct_prediction = tf.equal(y_pred_class, y_true_class_encoder)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        return (x_, y_), optimizer, accuracy, y_pred, cost, decay_learning_rate, cost_reg, keep_prob


    def optimize(self):
        acc_his = []
        cost_his = []
        for i in range(self.num_epochs):
            image_counter = 0
            print("")
            print("[INFO] Epoch #{0}".format(i+1))
            start_time = time.time()
            total_batch_num = int(len(self.trainImagePaths) / self.batch_size)
            bar_coeff =  21./total_batch_num
            # Average accuracy and cost of epoch across all batches
            acc_over_batch = 0.
            cost_over_batch = 0.
            for start, end in zip(range(0, len(self.trainImagePaths), self.batch_size), 
                                    range(self.batch_size, len(self.trainImagePaths), self.batch_size)):
                progress = int(start/self.batch_size)
                info = "[INFO] Batch #{0}/{1}".format(progress+1, total_batch_num)
                batch_imagePaths = self.trainImagePaths[start:end]
                (data, labels) = Network.getImagesLabels(self.img_size, batch_imagePaths)
                if data.shape[0] == 0:
                    continue
                labels_mat = Network.convert_to_categorical(labels, self.num_classes)
                feed_dict = {self.x_ : data, self.y_ : labels_mat, self.dropout_prob : self.dropout_probability,
                                self.is_training_pc:True}
                self.sess.run(self.train_step, feed_dict=feed_dict)
                info += "[" + "=" * int(progress * bar_coeff) + ">" + "-" * int((total_batch_num - progress)*bar_coeff) + "] "
                acc_b, cost_b, cost_rb = self.sess.run([self.accuracy, self.cost, self.cost_r], feed_dict=feed_dict)
                acc_his.append(acc_b)
                cost_his.append(cost_b)
                acc_over_batch += acc_b
                cost_over_batch += cost_b
                if end >= len(self.trainImagePaths)-self.batch_size:
                    acc_over_batch /= total_batch_num
                    cost_over_batch /= total_batch_num
                    info += "(Overall) Acc: {0:2.4%}".format(acc_over_batch)
                    info += ", Cost: {0:2.4f}".format(cost_over_batch)
                    info += " " * 6
                    print(info)
                else:
                    info += "Acc: {0:2.4%}".format(acc_b)
                    info += ", Cost: {0:2.4f}".format(cost_b)
                    info += ", w/ reg: {0:2.4f}".format(cost_rb)
                    print(info, end="\r")
            saver = tf.train.Saver()
            saver.save(self.sess, self.path_to_models)
            print("Learning rate: {0}".format(self.sess.run(self.dcl)))
            end_time = time.time()
            time_diff = end_time - start_time
            print("Dur: {0}".format(timedelta(seconds=time_diff)))
            self.validation_accuracy()
            plt.figure(1)
            plt.subplot(121)
            plt.plot(acc_his, 'g-', label="Accuracy")
            plt.xlabel("Training steps")
            plt.legend(loc="lower right")
            plt.subplot(122)
            plt.plot(cost_his, 'r-', label="Cost")
            plt.xlabel("Training steps")
            plt.legend(loc="upper right")
            if self.use_batchnorm:
                pylab.savefig("progress_cats_dogs_BN.png")
            else:
                pylab.savefig("progress_cats_dogs.png")


    def validation_accuracy(self):
        data = []
        labels = []
        for (i, imagePath) in enumerate(self.valImagePaths):
            image = cv2.imread(imagePath)
            image = cv2.resize(image, dsize=(self.img_size, self.img_size))
            label = imagePath.split(os.path.sep)[-1].split(".")[0]
            data.append(image)
            labels.append(label)
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        data = np.array(data) / 255.0
        if labels.shape[0] > self.batch_size:
            acc_total = 0
            for (start, end) in zip(range(0, labels.shape[0]+1, self.batch_size),
                                        range(self.batch_size, labels.shape[0]+1, self.batch_size)):
                batch_data = data[start:end]
                batch_labels = labels[start:end]
                batch_labels_mat = Network.convert_to_categorical(batch_labels, self.num_classes)
                acc = self.sess.run(self.accuracy, feed_dict={self.x_:batch_data, self.y_:batch_labels_mat,
                                                            self.dropout_prob:1., self.is_training_pc:False})
                acc_total += acc
            acc_avg = acc_total / (labels.shape[0] // self.batch_size)
            print("Validation accuracy: {0:.4}%".format(acc_avg))
        else:
            labels_mat = Network.convert_to_categorical(labels, self.num_classes)
            print("Validation accuracy: {0:.4%}".format(self.sess.run(self.accuracy, 
                                                        feed_dict={self.x_:data, self.y_:labels_mat, 
                                                        self.dropout_prob:1., self.is_training_pc:False})))

    def test_network(self):
        (x_, y_), _, accuracy, y_pred, _ , dcl, _, dropout_prob = self.build_graph(is_training=False)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, self.path_to_models)
        graph = tf.get_default_graph()
        # graph = tf.get_default_graph()
        # x_ = graph.get_tensor_by_name("x-input:0")
        # y_ = graph.get_tensor_by_name("y-labels:0")
        # y_pred = graph.get_tensor_by_name("y-predictions:0")
        # accuracy = graph.get_tensor_by_name("Accuracy:0")
        print("Variables:")
        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(i.name)
        (data, labels) = Network.getImagesLabels(self.img_size, self.testImagePaths)
        test_avg = graph.get_tensor_by_name("ConvLayer2/ConvLayer2/ConvLayer2_2/ConvLayer2/moments/normalize/mean/ExponentialMovingAverage:0")
        print(test_avg.eval(session=sess))
        # labels_mat = Network.convert_to_categorical(labels, self.num_classes)
        # acc, yp = sess.run([accuracy, y_pred], feed_dict={x_:data, y_:labels_mat, dropout_prob:1})
        test_image_counter = 0
        acc_total = 0
        test_batch_size = 150
        # if labels.shape[0] > test_batch_size:
        #     for (start, end) in zip(range(0, labels.shape[0]+1, test_batch_size), 
        #                             range(test_batch_size, labels.shape[0]+1, test_batch_size)):
        #         batch_data = data[start:end]
        #         batch_labels = labels[start:end]
        #         batch_labels_mat = Network.convert_to_categorical(batch_labels, self.num_classes)
        #         acc, yp = sess.run([accuracy, y_pred], feed_dict={x_:batch_data, y_:batch_labels_mat, 
        #                                                     dropout_prob:1., self.is_training_pc:False})
        #         print("Test ex {0}-{1}: {2:1.4f}".format(start, end, acc))
        #         test_image_counter += test_batch_size
        #         acc_total += acc
        #     batch_data = data[test_image_counter:]
        #     batch_labels = labels[test_image_counter:]
        #     batch_labels_mat = Network.convert_to_categorical(batch_labels, self.num_classes)
        #     acc, yp = sess.run([accuracy, y_pred], feed_dict={x_:batch_data, y_:batch_labels_mat, 
        #                                                     dropout_prob:1., self.is_training_pc:False})
        #     acc_avg = acc_total / (labels.shape[0] // test_batch_size)
        #     print("Average accuracy: {0:.4f}".format(acc_avg))
        true_counter = 0
        info = "."
        counter = 0
        for imagePath in self.testImagePaths:
            image_orig = cv2.imread(imagePath)
            image = cv2.resize(image_orig, dsize=(self.img_size, self.img_size))
            image = image / 255.
            image = np.expand_dims(image, axis=0)
            result = sess.run(y_pred, feed_dict={x_:image, dropout_prob:1, self.is_training_pc:False})
            label_str = imagePath.split(os.path.sep)[-1].split(".")[0]
            result_label = ""
            if counter % 50 == 0:
                info += "."
            counter += 1
            print(info, end="\r")
            if result[0,0] > result[0,1]:
                cv2.imshow('Cat', image_orig)
                result_label = "cat"
            else:
                cv2.imshow('Dog', image_orig)
                result_label = "dog"
            if result_label == label_str.lower():
                true_counter += 1
            print("Probs: {0}".format(result))
            k = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
        print("\nAcc: {0:0.5f}".format(true_counter/labels.shape[0]))
        print(result.shape)

    def get_fans(shape):
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[:-1])
        fan_out = shape[1] if len(shape) == 2 else shape[-1]
        return (fan_in, fan_out)

    def convert_to_categorical(labels, num_classes):
        new_labels = np.zeros((labels.shape[0], num_classes))
        for i in range(labels.shape[0]):
            new_labels[i, labels[i]] = 1
        return new_labels

    def overfit(num_iterations):
        for i in range(num_iterations):
            print("[INFO] Epoch #{0}".format(i+1))
            (data, labels) = getImagesLabels(self.img_size, imagePaths)
            labels_mat = convert_to_categorical(labels, num_classes)
            sess.run(optimizer, feed_dict={x_:data, y_:labels_mat})
            info = "\tAccuracy: {0:.4%}".format(sess.run(accuracy, feed_dict={x_:data, y_:labels_mat}))
            info += ", Cost: {0:.4}".format(sess.run(cost, feed_dict={x_:data, y_:labels_mat}))
            print(info, end="\r")


    # if isconv is true axes=[0,1,2] is applied to tf.nn.moments for convolutional layer, otherwise [0] 
    def batch_norm_wrapper(self, inputs, scope, is_training, decay=0.9):
        with tf.variable_scope(scope):
            epsilon = 1e-3
            shape = inputs.get_shape().as_list()
            # gamma: trainable scale factor
            gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0), 
                                        trainable=True)
            # beta: trainable shift value
            beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0),
                                        trainable=True)
            batch_mean, batch_var = tf.nn.moments(inputs, list(range(len(shape) - 1)))
            ema = tf.train.ExponentialMovingAverage(decay=decay)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
                    
            mean, var = tf.cond(self.is_training_pc, mean_var_with_update, 
                                    lambda : (ema.average(batch_mean), ema.average(batch_var)))
            return tf.nn.batch_normalization(inputs, mean, var, beta, gamma, epsilon)

            # if is_training:
            #     avg, var = tf.nn.moments(inputs, list(range(len(shape) - 1)) )
            #     update_moving_avg = moving_avg.assign(moving_avg * decay + avg * (1 - decay))
            #     update_moving_var = moving_var.assign(moving_var * decay + var * (1 - decay))
            #     control_inputs = [update_moving_avg, update_moving_var]
            # else:
            #     avg = moving_avg
            #     var = moving_var
            #     control_inputs = []
            # with tf.control_dependencies(control_inputs):
            #     return tf.nn.batch_normalization(inputs, avg, var, offset=beta, scale=gamma,
            #         variance_epsilon=epsilon)


        # return tf.contrib.layers.batch_norm(inputs, center=True, scale=True, is_training=is_training,
        #                                         scope=scope)
        # with tf.variable_scope(scope, reuse=True):
        #     gamma = tf.Variable(tf.ones([inputs.get_shape()[-1]]), name="gamma")
        #     beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), name="beta")
        #     pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), name="pop_mean", trainable=False)
        #     pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), name="pop_var", trainable=False)
        #     epsilon = 1e-4
        #     if is_training:
        #         if isconv:
        #             batch_mean, batch_var = tf.nn.moments(inputs, axes=[0,1,2])
        #         else:
        #             batch_mean, batch_var = tf.nn.moments(inputs, axes=[0])
        #         train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        #         train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        #         with tf.control_dependencies([train_mean, train_var]):
        #             print("Giving training mean, var to BN")
        #             return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon, name="BN")
        #     else:
        #         print("Giving pop_mean mean, var to BN")
        #         return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, epsilon, name="PopBN")


    def flatten_layer(layer, scope):
        with tf.variable_scope(scope):
            layer_shape = layer.get_shape()
            num_features = layer_shape[1:].num_elements()
            layer_flat = tf.reshape(layer, [-1, num_features], name="flatten")
        return layer_flat, num_features

    def getImagesLabels(img_size, imagePaths):
        data = []
        labels = []
        for imagePath in imagePaths:
            image = cv2.imread(imagePath)
            image = cv2.resize(image, dsize=(img_size, img_size))
            label = imagePath.split(os.path.sep)[-1].split(".")[0]
            data.append(image)
            labels.append(label)
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        labels = np.reshape(labels, (labels.shape[0], 1))
        data = np.array(data) / 255.
        return data, labels

if __name__ == '__main__':
    path_dict = {}
    path_dict['train'] = path_to_train_images
    path_dict['test'] = path_to_test_images
    path_dict['models'] = path_to_models
    network = Network(path_dict)
    
    is_training = 1
    network.is_training = 1
    args = vars(network.ap.parse_args())
    if args["istraining"] != None:
        is_training = args["istraining"]
    if is_training:
        print("Starting training..")
        network.train_network(contd=False)
    else:
        network.is_training = 0
        print("Starting testing..")
        network.test_network()
# print("Start time: ", datetime.datetime.now().time())
































