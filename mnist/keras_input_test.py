import sys
import numpy as np
np.random.seed(20160715)

from keras.datasets import mnist
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

for xs in X_train[0]:
    for x in xs:
        sys.stdout.write('%03d ' % x)
    sys.stdout.write('\n')

print('first sample is %d' % y_train[0])

Y_train = np_utils.to_categorical(y_train, 10)

sys.stdout.write('[')
for y in Y_train[0]:
    sys.stdout.write('%f ' % y)
sys.stdout.write(']\n')
