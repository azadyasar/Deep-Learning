from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import regularizers
import argparse


ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required=True)
# args = vars(ap.parse_args)

# print(args["dataset"])
train_data_dir = "/home/ec2-user/data/train_split"

img_width, img_height = 64, 64
nb_train_samples = 22000
epochs = 15
batch_size = 128
'''
if K.image_dim_ordering() == 'channel_first':
  print("Channel first")
  input_shape = (3, img_width, img_height)
else:
  print("Channel last")
  input_shape = (img_width, img_height, 3)
'''
input_shape = (img_width, img_height, 3)
print("Input shape: {0}".format(input_shape))

model = Sequential()
model.add(Conv2D(16, (3,3), input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3,3), kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512, kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(256, kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])


print(model.summary())

train_datagen = ImageDataGenerator(rescale=1./255.) #, shear_range=0.2, zoom_range=0.15,
  #horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(train_data_dir,
  target_size=(img_width,img_height), batch_size=batch_size,
  class_mode='categorical')

model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size,
  epochs=epochs, verbose=1)

model.save('model.h5')
