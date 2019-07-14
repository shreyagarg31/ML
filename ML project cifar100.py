from __future__ import absolute_import,division,print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras.activations
import keras.utils import to_categorical
cifar100=tf.keras.cifar100
(train_images,train_labels),(test_images,test_labels)=cifar100.load_data()

#new activation function

def talu(x):
    cond=tf.less_equal(x,x*0.0)
    t=tf.tanh(x)
    #chose value of hyperparam alpha as -.05
    tanH=tf.tanh(-0.05)
    cond1=tf.less_equal(x,-0.05*(1-x*0.0))
    y=tf.where(cond1,tanH*(1-x*0.0),t)
    return tf.where(cond,y,x)

keras.activations.custom_activation = talu

train_images=train_images.astype('float32')
test_images=test_images.astype('float32')

train_images=train_images/255.0
test_images=test_images/255.0
train_labels_hot=to_categorical(train_labels)
test_labels_hot=to_categorical(test_labels)
train_images,valid_images,train_labels,valid_labels=train_test_split(train_images,train_labels_hot,test_size=0.2)
#conv layers
model=tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation=talu,input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,padding='same',activation=talu))
model.add(BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,padding='same',activation=talu))
model.add(BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

model.add(tf.keras.layers.Conv2D(filters=384,kernel_size=3,padding='same',activation=talu))
model.add(BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

model.add(tf.keras.layers.Conv2D(filters=512,kernel_size=3,padding='same',activation=talu))
model.add(BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

model.add(tf.keras.layers.Conv2D(filters=768,kernel_size=3,padding='same',activation=talu))
model.add(BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

model.add(tf.keras.layers.Conv2D(filters=896,kernel_size=3,padding='same',activation=talu))
model.add(BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))


model.add(tf.keras.layers.Conv2D(filters=1024,kernel_size=3,padding='same',activation=talu))
model.add(BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

model.add(tf.keras.layers.Conv2D(filters=1152,kernel_size=3,padding='same',activation=talu))
model.add(BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))


model.add(tf.keras.layers.Flatten())
#first FC layer
model.add(tf.keras.layers.Dense(128,activation='linear'))
model.add(BatchNormalization())
model.add(activation('talu'))
model.add(tf.keras.layers.Dropout(0.3))
#second FC layer
model.add(tf.keras.layers.Dense(256,activation='linear'))
model.add(BatchNormalization())
model.add(activation('talu'))
model.add(tf.keras.layers.Dropout(0.4))
#third FC layer
model.add(tf.keras.layers.Dense(512,activation='linear'))
model.add(BatchNormalization())
model.add(activation('talu'))
model.add(tf.keras.layers.Dropout(0.4))
#fourth FC layer
model.add(tf.keras.layers.Dense(1024,activation='linear'))
model.add(BatchNormalization())
model.add(activation('talu'))
model.add(tf.keras.layers.Dropout(0.5))
#final FC
model.add(tf.keras.layers.Dense(100,activation='softmax'))


model.compile(optimizer='adam',loss='sparse_categorically_crossentropy',metrics=['accuracy'])
model.fit(train_images,train_labels,batch_size=128,epochs=40,verbose=1,validation_data=(valid_images,valid_labels))
test_loss,test_accuracy=model.evaluate(test_images,test_labels_hot,verbose=0)
print('test_accuracy:',test_accuracy)

          
          
