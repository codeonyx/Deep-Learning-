from keras.datasets import mnist

(X_train, y_train), (X_test, y_test)=mnist.load_data()

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

fig=plt.figure(figsize=(20,20))
for i in range(6):
    ax=fig.add_subplot(1,6,i+1)
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(y_train[i])
    

#View an image in more detail
def visualize_input(img, ax):
    ax.imshow(img, cmap='gray')
    width, height=img.shape
    thresh=img.max()/255
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy=(y, x), horizontalalignment='center', verticalalignment='center'
                        , color='white' if img[x][y]< thresh else 'black')


fig=plt.figure(figsize=(12, 12))
ax=fig.add_subplot(111)
visualize_input(X_train[0], ax)


#Rescaling the images by Dividing every pixel in every image by 255

X_train=X_train.astype('float32')/255
X_test=X_test.astype('float32')/255

#Encode categorical integer labels using one-hot scheme

from keras.utils import np_utils

print('Integer Valued Labels:')
print(y_train[:10])

y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)

print('One hot labels:')
print(y_train[:10])

#Define the model architecture
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

model=Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()


#Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#Calculate the classification accuracy on the test set

score=model.evaluate(X_test, y_test, verbose=0)
accuracy=100*score[1]

print('Test Accuracy: %.4f'%accuracy)

#Train the model 
from keras.callbacks import ModelCheckpoint

checkpointer=ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=1, save_best_only=True)

hist=model.fit(X_train, y_train, batch_size=128, epochs=100, validation_split=0.2,
               callbacks=[checkpointer], verbose=1, shuffle=True)


#Load the  model with the Best classification accuracy on the validation set
model.load_weights('mnist.model.best.hdf5')

#Calculate the classification accuracy on the test set
score=model.evaluate(X_test, y_test, verbose=0)

