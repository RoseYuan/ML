import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from keras.utils import to_categorical

# learning
learning_rate=5e-5
lambda_val = 0.05
epochs = 1
batch_size = 100 # batch_size for ann
# model params
IN_DIM = 139
NUM_CLASSES=10
encoder_units = [70, 50]
decoder_units = [70, IN_DIM]
ann_units = [50, 30, NUM_CLASSES]

labeled_train = pd.read_hdf("train_labeled.h5", "train")
unlabeled_train = pd.read_hdf("train_unlabeled.h5", "train")
test = pd.read_hdf("test.h5", "test")
labeled_y = labeled_train['y'].values
labeled_y = to_categorical(labeled_y)
labeled_x = labeled_train.iloc[:,1:].values
unlabeled_x = unlabeled_train.values
x_test = test.values
x_all = np.concatenate([labeled_x,unlabeled_x], axis=0)


## build graph
X = tf.placeholder(tf.float32,[None, IN_DIM])
Y = tf.placeholder(tf.float32,[None, NUM_CLASSES])
rep = X
# encoder
for nunits in encoder_units:
    rep = tf.layers.dense(rep, nunits, activation=tf.nn.relu)
# deconder
recon = rep
for nunits in decoder_units:
    recon = tf.layers.dense(recon, nunits, activation=tf.nn.relu)
# ANN
logits = rep
for nunits in ann_units:
    logits = tf.layers.batch_normalization(logits)
    logits = tf.layers.dense(logits, nunits, activation=tf.nn.relu)

L1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
L2 = tf.losses.mean_squared_error(X, recon)
L2 = lambda_val*L2

# loss = lambda_val*L2 + L1
optimizer = tf.train.AdamOptimizer(learning_rate)
train_L1 = optimizer.minimize(L1)
train_L2 = optimizer.minimize(L2)

# setup the initialisation operator
init_op = tf.global_variables_initializer()

# run session
with tf.Session() as sess:
    ## initialise the variables
    sess.run(init_op)

    batch_num = len(labeled_y)//batch_size+1 if len(labeled_y)%batch_size else len(labeled_y)//batch_size

    for epoch in range(1,epochs+1):
        k_fold = KFold(n_splits=batch_num, shuffle=True)

        for _,batch_index in k_fold.split(labeled_x):
            labeled_X = labeled_x[batch_index,:]
            labeled_Y = labeled_y[batch_index,:]
            L1_val, _ = sess.run([L1, train_L1], feed_dict={X: labeled_X, Y: labeled_Y})

        for _,batch_index in k_fold.split(x_all):
            all_X = x_all[batch_index,:]
            L2_val, _ = sess.run([L2, train_L2], feed_dict={X: all_X})

        # loss of the last batch
        print("epoch",epoch,"/",epochs,"L1: ", L1_val, "L2: ", L2_val)

    # predict
    ##output = sess.run(logits, {X: x_test})
    output = sess.run(tf.argmax(logits, 1), {X:x_test})
    print("The prediction for x_test is :", output)

