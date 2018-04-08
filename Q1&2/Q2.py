"""
Doeun Kim
CSE 353 HW1
"""

import tensorflow as tf
from string import punctuation
from sklearn.preprocessing import StandardScaler

digits, symbols, uppers, length, strength = [], [], [], [], []
with open('password1.train') as file:
    data = file.readlines()
    for line in data:
        line = line.split('\t')
        length.append([len(line[0])])
        strength.append([float(line[1].replace('\n', ''))])
        #strength.append([math.log10(float(line[1].replace('\n', '')))])
        num_digit, num_sym, num_up = 0, 0, 0
        for l in line[0]:
            if l.isdigit():
                num_digit += 1
            elif l in punctuation:
                num_sym += 1
            elif l.isupper():
                num_up += 1
        digits.append([num_digit])
        symbols.append([num_sym])
        uppers.append([num_up])

sc = StandardScaler()
digits, symbols, uppers, length, strength = sc.fit_transform(digits), sc.fit_transform(symbols), sc.fit_transform(uppers), sc.fit_transform(length), sc.fit_transform(strength)
W_digits = tf.Variable(tf.random_normal([1]), name='weight_digits')
W_symbols = tf.Variable(tf.random_normal([1]), name='weight_symbols')
W_uppers = tf.Variable(tf.random_normal([1]), name='weight_uppers')
W_length = tf.Variable(tf.random_normal([1]), name='weight_length')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypo = digits * W_digits + symbols * W_symbols + uppers * W_uppers + length * W_length + b
cost = tf.reduce_mean(tf.square(hypo - strength))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(201):
    sess.run(train)
    if step > 190:
        print('Step:', step, 'Cost:', sess.run(cost), 'Weight of number of digits:', sess.run(W_digits),
              'Weight of number of symbols:', sess.run(W_symbols), 'Weight of number of uppercase letters:', sess.run(W_uppers),
              'Weight of password lengths:', sess.run(W_length), 'Bias:', sess.run(b))