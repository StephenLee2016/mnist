import  input_data
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
'''
mnist 数据集由image和label两部分组成，image->x,label->y
训练集：mnist.train.images  标签：mnist.train.labels
每张图片都是28X28像素
'''

import tensorflow as tf
'''
初始一个二维变量x用来存储image, 每行是一张图片,
为一个长度为784的行向量
'''
x = tf.placeholder(tf.float32, [None, 784])
'''
定义模型的weight和bias
'''
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
'''
构建模型
'''
y = tf.nn.softmax(tf.matmul(x,W) + b)
'''
定义交叉熵函数
'''
y_ = tf.placeholder(tf.float32, [None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(150)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))