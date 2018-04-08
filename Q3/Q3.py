"""
Doeun Kim
CSE 353 HW1
"""
import tensorflow as tf
import math
from sklearn.preprocessing import StandardScaler

time = []
value = []
with open('export-EtherPrice.csv') as file:
    file.readline()
    data = file.readlines()
    for line in data:
        line = line.split(',')
        time.append([math.log(float(line[1].replace("\"", "")),10)])
        value.append([float(line[2].replace("\"", ""))])

sc = StandardScaler()
time,value = sc.fit_transform(time), sc.fit_transform(value)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypo = x * W + b
cost = tf.reduce_mean(tf.square(hypo - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(101):
    cur_cost, cur_W, cur_b, _ = sess.run([cost, W , b, train], feed_dict={x: time, y:value})
    if step > 90:
        print(step, cur_cost, cur_W, cur_b)

print("March 19th bitcoin price will be",sess.run(hypo, feed_dict={x:[1521450000]}))
