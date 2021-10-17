import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense,MaxPooling2D,Conv2D,Flatten,Input,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models,layers
import matplotlib.pyplot as plt
import pandas
import numpy
import os
from winsound import Beep


def plotAccuracyAndLoss(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy(haha sussy)')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss(SUSSY BAKA)')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

dataBatchSize=600



def preprocessData():
    #example line
    (Xtrain,Ytrain),(Xtest,Ytest)=keras.datasets.cifar10.load_data()
    assert Xtrain.shape == (50000, 32, 32, 3)
    assert Xtest.shape == (10000, 32, 32, 3)
    assert Ytrain.shape == (50000, 1)
    assert Ytest.shape == (10000, 1)
    
    Ytrain=keras.utils.to_categorical(Ytrain,10,dtype='int32')
    Ytest=keras.utils.to_categorical(Ytest,10,dtype='int32')
    
    imageDataGen=ImageDataGenerator(height_shift_range=0.25,width_shift_range=0.25,rotation_range=0.2,zoom_range=0.25,rescale=1.0/255,horizontal_flip=True)
    
    
    imageDataGen.fit(Xtrain)
    imageDataGen.fit(Xtest)
    
    trainingIterator=imageDataGen.flow(Xtrain,Ytrain,batch_size=dataBatchSize,shuffle=True)
    validationIterator=imageDataGen.flow(Xtest,Ytest,batch_size=dataBatchSize,shuffle=True)
    return trainingIterator,validationIterator,Xtrain,Ytrain

def defineModel():
    model = keras.Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.13))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.175))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.30))
    
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',metrics=['accuracy'])
    print(model.summary())

    return model
    



def trainModel(epochSize,model):
    trainingIterator,validationIterator,Xtrain,Xtest=preprocessData()
    history=model.fit(trainingIterator,
    batch_size=512,
    epochs=epochSize,
    validation_data=validationIterator,
    steps_per_epoch=len(Xtrain)/dataBatchSize,
    validation_steps=len(Xtest)/dataBatchSize)
    return history


def saveModel(model):
    model.save('cifar10FullModel')

def loadModel():
    return keras.models.load_model('cifar10FullModel')

def resumeTraining(epochSize,plotPerformance):
    model=loadModel()
    history=trainModel(epochSize,model)
    saveModel(model)
    if plotPerformance==True:
        plotAccuracyAndLoss(history)
    else:
        pass

def makeNewModelAndTrain(epochs,saveModelBool,plotPerformance):
    model=defineModel()
    history=trainModel(epochs,model)
    if saveModelBool:
        saveModel(model)
    else:
        pass
    
    if plotPerformance==True:
        plotAccuracyAndLoss(history)
    else:
        pass


for i in range(8):
    resumeTraining(1,False)