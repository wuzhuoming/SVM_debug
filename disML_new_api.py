import tensorflow as tf
import numpy as np
import logging
from tensorflow.python import debug as tf_debug

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

def parse_line_for_batch_for_libsvm(line):
    value = line.split(b' ')
    # logging.info(value)
    if value[0] == b'1':
        one_hot_label = [1, 0]
    else:
        one_hot_label = [0, 1]
    # logging.info(one_hot_label)
    label = value[0]
    indices = []
    values = []
    for item in value[1:]:
        [index, value] = item.split(b':')
        # if index start with 1, index = int(index)-1
        # else index=int(index)
        index = int(index) - 1
        value = float(value)
        indices.append(index)
        values.append(value)
    return label, one_hot_label, indices, values



def read_batch(sess, train_data_batch_tensor, batch_size):
    label_list = []
    ids = []
    sp_indices = []
    weight_list = []
    label_list_one_hot = []
    lines = sess.run(train_data_batch_tensor)
    for i, line in enumerate(lines):
        label, one_hot_label, indices, values = parse_line_for_batch_for_libsvm(line)
        label_list.append(label)
        label_list_one_hot.append(one_hot_label)
        ids += indices
        # sp_indices = np.array([[i, index] for index in indices])
        for index in indices:
            sp_indices.append([i, index])
        weight_list += values

    return np.reshape(label_list_one_hot, (batch_size, 2)), np.reshape(label_list, (
    batch_size, 1)), ids, sp_indices, weight_list, batch_size


def SVMModel_with_linear(x_data, y, num_features, variable_partition_num=108):
    # Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
    # L2 regularization parameter, alpha
    with tf.variable_scope('svm_parameter'):
        weight = tf.get_variable("weight", initializer=tf.constant(0.0, shape=[num_features, 1]),
                                 partitioner=tf.fixed_size_partitioner(variable_partition_num))
        #weight = tf.get_variable("weight", initializer=tf.constant(0.0, shape=[num_features, 1]))
        b = tf.Variable(tf.constant(0.1, shape=[1]))
        y_ = tf.subtract(tf.sparse_tensor_dense_matmul(x_data, weight), b)
        alpha = tf.constant([0.001])
    with tf.variable_scope('svm_loss'):
        l2_norm = tf.reduce_sum(tf.square(weight))
        classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(y_, y))))
        loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))
    return loss


def get_feed_dict(session,train_data_batch_tensor):
        """
        :param tf_program_util.TFProgramUtil context:
        :return:
        """
        sess = session
        batch_size = 2590
        label_one_hot, label, indices, sparse_indices, weight_list, read_count = read_batch(sess, train_data_batch_tensor,batch_size)
        return {y: label,
                sp_indices: sparse_indices,
                shape: [read_count,num_features],
                ids_val: indices,
                weights_val: weight_list}





num_features = 29890095
data_dir = "/research/d3/zmwu/model/svm_debug/data"

trainset_files = ["/research/d3/zmwu/model/svm_debug/data/kddb"]
train_filename_queue = tf.train.string_input_producer(trainset_files)
train_reader = tf.TextLineReader()
key_tensor, line_tensor = train_reader.read(train_filename_queue)
train_data_batch_tensor = tf.train.shuffle_batch([line_tensor],batch_size=2590,capacity=100,min_after_dequeue=50)



with tf.variable_scope('placeholder'):
    y = tf.placeholder(tf.float32, [None, 1])
    sp_indices = tf.placeholder(tf.int64, name="sp_indices")
    shape = tf.placeholder(tf.int64, name="shape")
    ids_val = tf.placeholder(tf.int64, name="ids_val")
    weights_val = tf.placeholder(tf.float32, name="weights_val")

with tf.variable_scope('parameter'):
    x_data = tf.SparseTensor(sp_indices, weights_val, shape)


with tf.variable_scope('loss'):
    SVM_loss = SVMModel_with_linear(x_data, y, num_features)

train_op = tf.train.AdamOptimizer(learning_rate=0.00008).minimize(SVM_loss)

logging.info("start training here")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)


    for i in range(4000):
        logging.info("run data prepare for epoch %d" % i)
        fd = get_feed_dict(sess,train_data_batch_tensor)
        logging.info("feed dict generate finish.")

        logging.info("start Epoch %d" % i)
        
        
        _,loss = sess.run([train_op,SVM_loss], feed_dict=fd)
        logging.info(loss)

    # if i % 10 == 0:
    #     print("Loss: ", SVM_loss)
        logging.info("Epoch Done")
