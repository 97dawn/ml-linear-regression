"""
Doeun Kim
CSE 353 HW1
"""
import tensorflow as tf
import math
from sklearn.preprocessing import StandardScaler

pwd_length = []
pwd_strength = []

# pre-processing
with open('password1.train') as file:
    data = file.readlines()
    for line in data:
        len_a , stren_a = [], []
        line = line.split('\t')
        len_a.append(len(line[0]))
        stren_a.append(float(line[1].replace('\n', '')))
        #stren_a.append(math.log10(float(line[1].replace('\n', ''))))
        pwd_strength.append(stren_a)
        pwd_length.append(len_a)

sc = StandardScaler()
pwd_length = sc.fit_transform(pwd_length)
pwd_strength = sc.fit_transform(pwd_strength)
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypo = pwd_length * W + b
cost = tf.reduce_mean(tf.square(hypo - pwd_strength))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(201):
    sess.run(train)
    if step > 190:
        print('Step:',step,'Cost:', sess.run(cost),'Weight:', sess.run(W),'Bias:', sess.run(b))


