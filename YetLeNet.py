import tensorflow.examples.tutorials.mnist.input_data as mnist_input_data
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import pandas as pd
import csv

KEEP_PROB = 0.5
LEARNING_RATE = 1.0e-4 
EPOCH = 10 #一次
BATCH_SIZE = 32 #样本数
MODEL_FILE_NAME = 'model/model.save'
acc=0


class LeNet:
    def __init__(self):
        self.raw_input_image = tf.placeholder(tf.float32, [None, 784])
        self.input_images = tf.reshape(self.raw_input_image, [-1, 28, 28, 1])
        self.raw_input_label = tf.placeholder('float', [None, 10])
        self.input_labels = tf.cast(self.raw_input_label,tf.int32)
        self.drop_out = KEEP_PROB

        with tf.variable_scope('LeNet') as scope:
            self.train_net = self.construct_net(True)
            scope.reuse_variables() #变量重用
            self.pred_net = self.construct_net(False)

        self.prediction = tf.argmax(self.pred_net, 1)
        self.label = tf.equal(self.prediction, tf.argmax(self.input_labels, 1))
        self.train_accuracy = tf.reduce_mean(tf.cast(self.label, 'float'))
        self.loss = slim.losses.softmax_cross_entropy(self.train_net, self.input_labels)
        self.learning_rate = LEARNING_RATE
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def construct_net(self,is_training = True):
        with slim.arg_scope([slim.conv2d], padding='VALID',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.conv2d(self.input_images,6,[5,5],1,padding='SAME',scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.conv2d(net,16,[5,5],1,scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.conv2d(net,120,[5,5],1,scope='conv5')
            net = slim.flatten(net, scope='flat6')
            net = slim.fully_connected(net, 84, scope='fc7')
            net = slim.dropout(net, self.drop_out,is_training=is_training, scope='dropout8')
            output = slim.fully_connected(net, 10, scope='fc9')
        return output

def main(): 
    mnist = mnist_input_data.read_data_sets('data/', one_hot=True)
    batch_size = BATCH_SIZE
    parameter_path = MODEL_FILE_NAME
    lenet = LeNet()
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        if os.path.exists(parameter_path):
            saver.restore(parameter_path)
        else:
            sess.run(tf.initialize_all_variables())        
        for i in range(len(mnist.train.images) * EPOCH // BATCH_SIZE):
            batch = mnist.train.next_batch(BATCH_SIZE)
            if i % 100 == 0:
                eval_batch = mnist.test.next_batch(BATCH_SIZE)
                train_accuracy = sess.run(lenet.train_accuracy,
                                          feed_dict={
                                          lenet.raw_input_image: eval_batch[0],
                                          lenet.raw_input_label: eval_batch[1] }
                                          )
                print('step {}, training accuracy {:.4f}'.format(i, train_accuracy))
                acc=train_accuracy
            sess.run(lenet.train_op,feed_dict={lenet.raw_input_image: batch[0],lenet.raw_input_label: batch[1]})
        
        save_path = saver.save(sess, parameter_path)

    csvfile = open('1.csv','a')  
    writer = csv.writer(csvfile)

    data = [
        ('KEEP_PROB', KEEP_PROB),
        ('LEARNING_RATE', LEARNING_RATE),
        ('EPOCH',EPOCH),
        ('BATCH_SIZE',BATCH_SIZE),
        ('train_accuracy',acc)
    ]
    writer.writerows(data)
    csvfile.close()

if __name__ == '__main__':
    main()
