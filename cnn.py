import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import pandas as pd
import matplotlib.pyplot as plt

batch_size = 64
num_classes = 10
epochs = 12

img_rows, img_cols = 28, 28
train = pd.read_csv("./input/train.csv")

test_images = (pd.read_csv("./input/test.csv").values).astype('float32')
train_images = (train.ix[:,1:].values).astype('float32')
train_labels = train.ix[:,0].values.astype('int32')

if K.image_data_format() == 'channels_first':
    train_images = train_images.reshape(train_images.shape[0], 1, 28, 28)
    test_images = test_images.reshape(test_images.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

# Feature Standardization
train_images = train_images / 255
test_images = test_images / 255

train_labels = keras.utils.to_categorical(train_labels)
num_classes = train_labels.shape[1]

# Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(train_images, train_labels, validation_split=0.05, epochs=epochs, batch_size=batch_size)

predictions = model.predict_classes(test_images, verbose=0)
submissions = pd.DataFrame({"ImageId":list(range(1, len(predictions)+1)),
                            "Label":predictions})
submissions.to_csv("result.csv", index=False, header=True)
