import os
import tensorflow as tf
tf.random.set_seed(10)
import numpy as np

from csen_regressor.utils import *

class model:
    def __init__(self):
        ##change image size M depending on model: for InceptionV3 and Xception 12, for DenseNet121 ,VGG19 15, ResNet50 13
        ##adjust input size size here too! depending on which model is used
        ##(MxN :: 12x60 for Xception and Inception; 15x60 VGG; 39x20 ResNet; 27x20 for DenseNet)
        self.imageSizeM = 39
        self.imageSizeN = 20
        self.model = None
        self.history = None
        self.x_train = None
        self.x_val = None
        self.x_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def loadData(self, feature_type, set, MR):
        # Check if the CSEN data is available.
        if not os.path.exists('CSENdata-2D/'): exit('CSENdata-2D is not prepared!')
        data = 'CSENdata-2D/' + feature_type
        dataPath = data + '_mr_' + MR + '_run' + str(set) + '.mat'
        dic_label = scipy.io.loadmat('CSENdata-2D/dic_label' + '.mat')["ans"]

        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = loadData(dataPath)
        print(self.x_train.shape)

    def getModel(self):
        input_shape = (self.imageSizeM, self.imageSizeN, 1)

        input = tf.keras.Input(input_shape, name='input')
        ##change x_0 = tf.keras.layers.Conv2D(128, 5, padding = 'same', activation = 'relu')(input)
        ##for InceptionV3 and Xception model! not for other models
        x_0 = tf.keras.layers.Conv2D(64, 5, padding = 'same', activation = 'relu')(input)
        x_0 = tf.keras.layers.MaxPooling2D(pool_size=(5, 4))(x_0) # Sparse code shapes.
        x_0 = tf.keras.layers.Conv2D(1, 5, padding = 'same', activation = 'relu')(x_0)
        
        
        y = tf.keras.layers.Flatten()(x_0)
        y = tf.keras.layers.Dense(1, activation = 'softplus')(y)
        
        self.model = tf.keras.models.Model(input, y, name='CSEN')
        self.model.summary()

    def train(self, weightPath, epochs = 100, batch_size = 16):
        adam = tf.keras.optimizers.Adam(
            lr=0.001, beta_1=0.9, beta_2=0.999, 
            epsilon=None, decay=0.0, amsgrad=True)

        checkpoint_csen = tf.keras.callbacks.ModelCheckpoint(
            weightPath, monitor='val_loss', verbose=1,
            save_best_only=True, mode='min')
        
        callbacks_csen = [checkpoint_csen]
        
        ##changed the while loop. excluded cnvergence criteria and just have 5 runs with 100 epochs instead. If model works convergence is fine. to just test not suitable --> fails if model not converging no break?
        
        #while True:
        self.getModel()
        self.model.compile(loss = tf.compat.v1.losses.huber_loss,
        optimizer = adam, metrics=['mae', 'mse'] )
        # Training.
        print(self.x_train.shape)
        print(self.y_train.shape)
        self.history = self.model.fit(self.x_train, self.y_train,
        validation_data=(self.x_val, self.y_val),
        epochs = epochs, batch_size = batch_size,
        shuffle = True, callbacks=callbacks_csen)

            #if self.model.history.history['loss'][1] < 9: # If it is converged.
             #   break

    def load_weights(self, weightPath):
        self.getModel()
        self.model.load_weights(weightPath)

    def predict(self):
        y_pred = self.model.predict(self.x_test)
        return y_pred
######
