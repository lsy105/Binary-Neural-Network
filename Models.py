from binary_blocks import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()
model.add(Binary_Conv2D(32, 3, pad=1, binarize_input=True))
model.add(HardTanh())
model.add(Binary_Conv2D(32, 3))
model.add(HardTanh())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Binary_Conv2D(64, 3, pad=1))
model.add(HardTanh())
model.add(Binary_Conv2D(64, 3))
model.add(HardTanh())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Binary_Dense(512))
model.add(HardTanh())
model.add(Dropout(0.5))
model.add(Binary_Dense(10))
model.add(Activation('softmax'))
