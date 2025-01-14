import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.io
import tensorflow as tf
tf.random.set_seed(10)
import os
import sys
sys.path.append('../')
from csen_regressor import model
import argparse
from sklearn.model_selection import train_test_split


# INITIALIZATION
# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('--method', default='CSEN', 
                help="Method for the regression: CL-CSEN, CSEN, CL-CSEN-1D, CSEN-1D, SVR.")
ap.add_argument('--feature_type', help="Features extracted by the network (DenseNet121, VGG19, ResNet50).")
ap.add_argument('--weights', default=False, help="Evaluate the model.")
args = vars(ap.parse_args())
modelType = args['method'] # CL-CSEN, CSEN, and SVR.
feature_type = args['feature_type']
weights = args['weights']


MR = '0.5' # Measurement rate for CL-CSEN and CSEN approaches.

if modelType == 'CL-CSEN':
    from cl_csen_regressor import model
elif modelType == 'CL-CSEN-1D':
    from cl_csen_1d_regressor import model   
elif modelType == 'CSEN':
    from csen_regressor import model  
elif modelType == 'CSEN-1D':
    from csen_1d_regressor import model  
elif modelType == 'SVR':
    from competing_regressor import svr as model  

# From where to load weights
weightsDir = '../weights/' + modelType + '/'
   
# Init the model
modelFold = model.model()
weightPath = weightsDir + feature_type + '_' + MR + '_1' + '.h5'    
modelFold.load_weights(weightPath)

# Load image to be evaluated
data = '../CSENdata-2D/' + feature_type
dataPath = data + '_mr_' + MR + '_run1'  + '.mat'

dic_label = scipy.io.loadmat('../CSENdata-2D/dic_label' + '.mat')["ans"]
x_train, X_val, x_test, y_train, y_val, y_test =  None, None, None, None, None, None 
Data = scipy.io.loadmat(dataPath)

x_dic = Data['x_dic'].astype('float32')
x_train = Data['x_train'].astype('float32')
x_test = Data['x_test'].astype('float32')
y_dic = Data['dicRealLabel'].astype('float32')
y_train = Data['trainRealLabel'].astype('float32')
y_test = Data['testRealLabel'].astype('float32')


print('\n\n\n')
print('Loaded dataset:')
print(len(x_train), ' train')
print(len(x_test), ' test')

# Partition for the validation.
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 1)

# Data normalization.
m =  x_train.shape[1]
n =  x_train.shape[2]

x_dic = np.reshape(x_dic, [len(x_dic), m * n])
x_train = np.reshape(x_train, [len(x_train), m * n])
x_val = np.reshape(x_val, [len(x_val), m * n])
x_test = np.reshape(x_test, [len(x_test), m * n])

scaler = StandardScaler().fit(np.concatenate((x_dic, x_train), axis = 0))
x_dic = scaler.transform(x_dic)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

x_dic = np.reshape(x_dic, [len(x_dic), m, n])
x_train = np.reshape(x_train, [len(x_train), m, n])
x_val = np.reshape(x_val, [len(x_val), m, n])
x_test = np.reshape(x_test, [len(x_test), m, n])

x_train = np.concatenate((x_dic, x_train), axis = 0)
y_train = np.concatenate((y_dic, y_train), axis = 0)


print("\n")
print('Partitioned.')
print(len(x_train), ' Train')
print(len(x_val), ' Validation')
print(len(x_test), ' Test\n')
print("\n\n\n")

x_train = np.expand_dims(x_train, axis=-1)
x_val = np.expand_dims(x_val, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)


print("x_test shape: {}".format(x_test.shape))
print("Image shape: {}".format(x_test[0:2].shape))
image = x_test[0:30,:,:,:]

print("Image shape: {}".format(image.shape))

y_pred = modelFold.predict_distance(image)

print("y_pred: {}".format(y_pred))

print("vs.")
print("y_test: {}".format(y_test[0:20]))