from __future__ import absolute_import,division,print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
mnist=tf.keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
train_images=train_images.reshape(-1,28,28,1)
test_images=test_images.reshape(-1,28,28,1)
train_images=train_images/255.0
test_images=test_images/255.0
model=tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=2,padding='same',activation='relu',input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=2,padding='same',activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=10)
test_loss,test_acc=model.evaluate(test_images,test_labels)
print('test accuracy:',test_acc)
def plot_image(i,predictions_array,true_y,x):
    predictions_array,true_y=predictions_array[i],true_y[i]
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(x.reshape(28,28),cmap='gray',interpolation='none')
    
    predicted_label=np.argmax(predictions_array)
    if predicted_label==true_y:
        color='blue'
    else:
        color='red'
        
    plt.xlabel("predicted:{} {:2.0f}% (Truth: {})".format(predicted_label,100*np.max(predictions_array),true_y,color=color))
    predictions=model.predict(test_images)
    i=0
plot_image(i,predictions,test_labels,test_images[i])
