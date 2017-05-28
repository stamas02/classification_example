from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from log import Log



batch_size = 128
epochs = 12
#you can choose any two number between 0 and 9
numbers_to_classify = [2,7]
num_classes = len(numbers_to_classify)

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#Remove all the other numbers form the thest and training set but the two we want to classify
idxs = np.argwhere( np.logical_or( y_train == numbers_to_classify[0], y_train == numbers_to_classify[1]) )
y_train = y_train[idxs]
x_train = x_train[idxs]
idxs = np.argwhere( np.logical_or( y_test == numbers_to_classify[0], y_test == numbers_to_classify[1]) )
y_test = y_test[idxs]
x_test = x_test[idxs]

#re-labeling the two class. If you chose to callify 2 and 7 then the labels should be 0 and 1.
unique_classes = np.unique(y_train)
for u in range(len(unique_classes)):
    y_train[y_train==unique_classes[u]] = u
    y_test[y_test==unique_classes[u]] = u



if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

log = Log(log_dir = "logs/")
log.set_model(model)

for epoch in range(epochs):
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=1,
                        verbose=1,
                        validation_data=(x_test, y_test))

    """We use the training set as a validation set as well now. It is good practice to have a separate training, validation and test set
    if you have a separate test set then uncomment the ffollowing line to test the preformance on the testset"""
    #score = model.evaluate(x_test, y_test, verbose=0) 
    print('Test loss:', history.history['val_loss'][0])
    print('Test accuracy:', history.history['val_acc'][0])
    stat = {}
    stat['Test loss'] = history.history['val_loss'][0]
    stat['Test accuracy:'] = history.history['val_acc'][0]
    stat['Training loss'] = history.history['loss'][0]
    stat['Training acc'] = history.history['acc'][0]
    log.log(model, {}, stat, epoch)
