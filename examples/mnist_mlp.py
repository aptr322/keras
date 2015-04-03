import gzip
import cPickle

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils


def load_data(dataset):
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    return [train_set, valid_set, test_set]



print "loading data"

(x_train_set, y_train_set), (x_valid_set, y_valid_set), (x_test_set, y_test_set) = load_data('mnist.pkl.gz')

print "training_set: ",x_train_set.shape," ",y_train_set.shape

y10_train = np_utils.to_categorical(y_train_set, 10)
y10_test = np_utils.to_categorical(y_test_set, 10)


print "Building model..."
model = Sequential()
model.add(Dense(784, 1200, init='normal'))
model.add(Activation('relu'))
#model.add(Dense(1200,1200, init='normal'))
#model.add(Activation('relu'))
#model.add(BatchNormalization(input_shape=(800,)))
model.add(Dropout(0.5))
model.add(Dense(1200, 10, init='normal'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=0.0, momentum=0.2, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd)


print "Training..."
for step in range(0, 40):
    print "step:", step
    model.fit(x_train_set, y10_train, nb_epoch=1, batch_size=40)

    score = model.evaluate(x_test_set, y10_test, batch_size=200)
    print 'Test score:', score


    test_result = model.predict_classes(x_test_set)
    num_errors = np.count_nonzero(y_test_set - test_result)

    print "Num test errors: ", num_errors, " ", float(num_errors)/y_test_set.shape[0] * 100.

