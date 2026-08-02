import os

import numpy as np
import pandas as pd
import tensorflow as tf

####### Prepare data set ##########
csv_dir = os.path.dirname(os.path.abspath(__file__))  # folder of this script
df = pd.read_csv(os.path.join(csv_dir, 'Layer.csv'), dtype=np.float32)
data_X = df[['X1', 'X2', 'X3']].values  # training input

# correct answers will go here
data_Y = df['X1 or X2 xor X3'].values.reshape(-1, 1)  # convert to the vector
##################################

####### Create neural network architecture with TensorFlow 2 ##########
tf.random.set_seed(0)

totalFeatures = data_X.shape[1]  # column number (in this example: 3)
L = 5  # Number of nodes (neurons) of layer1
M = 1  # Number of nodes (neurons) of layer2

X = tf.constant(data_X)   # 8 x 3 (number_samples x 3)
Y_ = tf.constant(data_Y)  # 8 x 1 (number_samples x 1)

# Declare variables (weight and bias) that are updated when training the model
W1 = tf.Variable(tf.random.truncated_normal([totalFeatures, L], stddev=0.1))
B1 = tf.Variable(tf.zeros([L]))
W2 = tf.Variable(tf.random.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.zeros([M]))


def forward(X):
    # 2 fully connected layers
    L1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)   # output: 8 x 5 (number_samples x L)
    return tf.nn.sigmoid(tf.matmul(L1, W2) + B2)  # prediction output, 8 x 1


# use gradient descent algorithm to update weight and bias (W1, W2, B1, B2)
learning_rate = 0.1
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

##### Training here ##########
for step in range(30000):  # backpropagation training
    with tf.GradientTape() as tape:
        Predict = forward(X)
        loss = tf.reduce_mean(tf.square(Predict - Y_))  # loss = Predict - correct answers

    gradients = tape.gradient(loss, [W1, B1, W2, B2])
    optimizer.apply_gradients(zip(gradients, [W1, B1, W2, B2]))

    # If pred is more than 0.5, the answer is 1
    # If pred is less than 0.5, the answer is 0
    # In python, True is 1 and False is 0
    pred = 1 * (Predict.numpy() > 0.5)

    # accuracy of the trained model
    accuracy = np.mean(1 - np.abs(pred - data_Y)) * 100
    if accuracy == 100:  # finish
        print('\nTraining accuracy: %.1f%%' % accuracy)
        print('Finish learning at step: %d' % step)
        break

    if step % 3000 == 0:  # for debug
        print('\nTraining accuracy: %.1f%%' % accuracy)
        print('Loss at step %d: %f' % (step, float(loss)))

w1, b1 = W1.numpy(), B1.numpy()
w2, b2 = W2.numpy(), B2.numpy()
##############################

### for test #####
if __name__ == '__main__':
    print('\nAll input/output dataset')
    print(df)

    X1 = df['X1']
    X2 = df['X2']
    X3 = df['X3']

    # X1 or X2
    temp = np.logical_or(X1, X2)
    # (X1 or X2) xor X3
    result = np.logical_xor(temp, X3)
    print('\nLogic result: X1 or X2 xor X3')
    print(result)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    new_X = [0, 1, 1]  # input for testing

    # Calculate the logic: (X1 or X2) xor X3
    Ouput_layer1 = sigmoid(np.matmul(new_X, w1) + b1)
    Ouput_layer2 = sigmoid(np.matmul(Ouput_layer1, w2) + b2)

    print('\nFor testing')
    print('\nInput:', new_X)
    print('\nWeight for layer 1:\n', w1)
    print('\nWeight for layer 2:\n', w2)
    print('\nOutput values of layer 1:\n', Ouput_layer1)
    print('\nOutput values of layer 2:\n', Ouput_layer2)

    answer = 1 * (Ouput_layer2 > 0.5)
    print('\nAnswer is ', answer[0])
