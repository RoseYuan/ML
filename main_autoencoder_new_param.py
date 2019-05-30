'''author: SY'''

from keras.layers import Dense,Dropout,BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.utils import to_categorical
from keras import optimizers

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


# (hyper) parameters
## preprocessing
standardize = True
do_shuffle = False
## pre train
pre_train = True
epochs_pre = 100
batch_size_pre = 100
## train
dump_model = True
validate = True
epochs = 200
batch_size = 100
validation_size = 0.1
dropout_rate = [0.5,0.3,0.2,0.05]
using_test = False
## train with pseudo label
pred_unlabeled = False

# # fix random seed for reproducibility
# seed = 7
# np.random.seed(seed)
# prng = np.random.RandomState(seed)
# PYTHONHASHSEED=seed



train_labeled = pd.read_hdf("train_labeled.h5", "train")
train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
test = pd.read_hdf("test.h5", "test")

# shuffle data ??because fit does not shuffle for you?? really???
if do_shuffle:
    print("Shuffling...")
    train_labeled = shuffle(train_labeled, random_state=prng)
    train_unlabeled = shuffle(train_unlabeled, random_state=prng)

y = train_labeled['y'].values
y = to_categorical(y)  # Convert labels to categorical one-hot encoding
x_labeled = train_labeled.iloc[:,1:].values
x_unlabeled = train_unlabeled.values
x_test = test.values

# using test data as unlabeled data for training
if using_test:
    x_unlabeled = np.concatenate([x_unlabeled,x_test],axis=0)

x_all = np.concatenate([x_labeled,x_unlabeled], axis=0)

input_d = x_labeled.shape[1]
output_d = y.shape[1]



# standardize on the combination of labeled and unlabeled data
if standardize:
    print("standardizing")
    scaler=StandardScaler()
    x_all = scaler.fit_transform(x_all)
    x_labeled = scaler.transform(x_labeled)
    x_unlabeled = scaler.transform(x_unlabeled)
    x_test = scaler.transform(x_test)



def create_autoencoder():
    model = Sequential()
    # encoder
    model.add(Dense(960,input_dim=input_d,activation='relu',kernel_initializer='random_uniform'))
    #model.add(BatchNormalization())
    #model.add(Dropout(rate=dropout_rate[0], seed=seed))
    model.add(Dense(480, activation='relu',kernel_initializer='random_uniform'))
    #model.add(BatchNormalization())
    #model.add(Dropout(rate=dropout_rate[1], seed=seed))
    model.add(Dense(240, activation='relu',kernel_initializer='random_uniform'))
    #model.add(BatchNormalization())
    #model.add(Dropout(rate=dropout_rate[2], seed=seed))
    model.add(Dense(72, activation='relu',kernel_initializer='random_uniform'))
    #model.add(BatchNormalization())
    #model.add(Dropout(rate=dropout_rate[3], seed=seed))
    #model.add(Dense(output_d, activation='softmax'))

    # decoder
    model.add(Dense(72, activation='relu',kernel_initializer='random_uniform'))
    #model.add(Dense(48, activation='relu'))
    #model.add(Dense(96, activation='relu'))
    model.add(Dense(input_d, activation='linear',kernel_initializer='random_uniform'))

    model.compile(loss='mse', optimizer='adadelta')
    return model

# unsupervised pre-train
if pre_train:
    pre_train_model = create_autoencoder()
    pre_train_model.fit(x_unlabeled, x_unlabeled,
                    epochs=epochs_pre,
                    batch_size=batch_size_pre,
                    validation_data=(x_test, x_test))
    layer_weights = []
    for layer in pre_train_model.layers:
        single_layer_weights = layer.get_weights()
        if len(single_layer_weights) == 2:
            layer_weights.append(K.variable(single_layer_weights[0]))
else:
    layer_weights=[]

# test if the weight has been correctly passed
# for layer in pre_train_model.layers:
#     single_layer_weights = layer.get_weights()
#     if len(single_layer_weights) == 2:
#         print(single_layer_weights[0])

# supervised training

## custom initializers (Is there any elegant way to do this with Keras ???)

def my_inits(lw_index,lw=layer_weights,p_train=pre_train):
    if p_train:
        def my_init_i(shape):
            return K.variable(lw[lw_index])
        return my_init_i
    else:
        return "random_uniform"


def create_model():
    model = Sequential()
    # the kernel_initializer must be a string or callable
    model.add(Dense(960,input_dim=input_d,activation='relu',kernel_initializer=my_inits(0,lw=layer_weights,p_train=pre_train)))
    model.add(Dropout(rate=dropout_rate[0]))
    model.add(BatchNormalization())

    model.add(Dense(480, activation='relu',kernel_initializer=my_inits(1,lw=layer_weights,p_train=pre_train)))
    model.add(Dropout(rate=dropout_rate[1]))
    model.add(BatchNormalization())

    model.add(Dense(240, activation='relu',kernel_initializer='random_uniform'))
    model.add(Dropout(rate=dropout_rate[2]))
    model.add(BatchNormalization())

    model.add(Dense(60, activation='relu',kernel_initializer='random_uniform'))
    model.add(Dropout(rate=dropout_rate[3]))
    model.add(BatchNormalization())

    model.add(Dense(10, activation='softmax',kernel_initializer='random_uniform'))
    optim = optimizers.Adadelta()
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    return model

if dump_model:
    model = create_model()
    model.fit(x_labeled,
              y,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=validation_size if validate else 0)

    ## test if the weight has been correctly passed
    # for layer in model.layers:
    #     single_layer_weights = layer.get_weights()
    #     if len(single_layer_weights) == 2:
    #         print(single_layer_weights[0])

    y_predict = np.argmax(model.predict(x_test), axis=1)


    results = pd.read_csv('sample.csv')
    results['y'] = y_predict
    results.to_csv('sample.csv',index=False,header=True)

    if pred_unlabeled:
        # Predict unlabelled
        # Select best class for each sample with argmax
        y_unlabeled_pred = np.argmax(model.predict(x_unlabeled), axis=1)

        ## Add unlabel prediction to training
        y_unlabeled_pred_cat = to_categorical(y_unlabeled_pred)
        y_new = np.vstack((y, y_unlabeled_pred_cat))

        ## Create a new model
        #model_new = create_model()

        ## Fit the trained model with new data
        model_new = model
        model_new.fit(x_all,
                      y_new,
                      epochs=epochs,
                      batch_size=batch_size,

                      validation_split=validation_size if validate else 0,
                      callbacks=[EarlyStopping(monitor='val_loss',
                                               min_delta=0,
                                               patience=10)] if validate else []
                      )

        # Predict on test set
        y_pred = np.argmax(model_new.predict(x_test), axis=1)

        results1 = pd.read_csv('sample1.csv')
        results1['y'] = y_pred
        results1.to_csv('sample1.csv',index=False,header=True)