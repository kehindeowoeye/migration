import keras
from keras.layers import Input, LSTM, Dense, GRU
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
import numpy.matlib
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
import pandas as pd
import json
from keras.layers import Dense, Activation
import math
from keras.layers import TimeDistributed
import xlrd
import xlsxwriter
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

SEED = 102
#import os
#os.environ['PYTHONHASHSEED']=str(SEED)

import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.set_random_seed(SEED)

from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


"""
def AB_train(Xtrain,ytrain):
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(Xtrain, ytrain)
    return clf

def AB_test(clf, Xtest,ytest):
    ba = clf.predict(Xtest)
    acc = np.abs(ba - ytest)
    acd = 1-(np.count_nonzero(acc)/len(ytest))
    return acd
 

def LR_train(Xtrain,ytrain):
    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(Xtrain, ytrain)
    return clf

def LR_test(clf, Xtest,ytest):
    ba  = clf.predict(Xtest)
    acc = np.abs(ba - ytest)
    print("i am looking ")
    print(acc)
    print(acc.shape)
    acd = 1-(np.count_nonzero(acc)/len(ytest))
    return acd
"""

look_back = 50
num_features = 28
num_class = 4
num_classes = num_class








##########################################################################
data = np.array(pd.read_excel('new_vulture_data.xlsx'))
data = pd.DataFrame(data)
data = data.fillna(method='ffill')
data = np.array(data)

#disney, mac,sarkis,morongo,rosalie
rosalie = data[data[:,13]=='Rosalie']


rosalie_label = rosalie[:,11]
rosalie_input  = rosalie[:,2:4]
sec_inp = rosalie_input[:,0]
#sec_inp = rosalie[:,0]

rosalie= rosalie[:,16:]
#sec_inp = rosalie[:,25]
print(sec_inp)
rosalie_input  =np.hstack((rosalie_input, rosalie))


no = 24# hours in advance
print(len(rosalie_label))
print(len(rosalie_label[no:len(rosalie_label)] ) )
rosalie_label = rosalie_label[no:len(rosalie_label)]
rosalie_input = rosalie_input[0:len(rosalie_input)-no]
sec_inp = sec_inp[no:len(sec_inp)]
print(len(rosalie_input))

"""
ad = rosalie_label.astype(int)
clf =  AB_train(rosalie_input,ad)
acc = AB_test(clf,rosalie_input,ad)
print(acc)
"""


workbook = xlsxwriter.Workbook('rosalie_label_oneday.xlsx')
worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(rosalie_label.reshape(len(rosalie_label),1).T):
    worksheet.write_column(row, col, data)
workbook.close()


nb_samples = rosalie_input.shape[0] - look_back
Xtrain2 = np.zeros((nb_samples,look_back,num_features))
y_train_reshaped2 = np.zeros((nb_samples,1,num_class))
one_hot_labels2 = np.zeros((nb_samples,1,num_class))
pd_label = np.zeros((nb_samples,1,1))
ytra = np.array(pd.get_dummies(np.array(rosalie_label.astype(int).reshape(-1))))

for i in range(nb_samples):
    y_position = i + look_back
    Xtrain2[i] = rosalie_input[i:y_position]
    one_hot_labels2[i] = ytra[y_position,:num_class]
    pd_label[i] = sec_inp[y_position]
rosalie_input = Xtrain2
rosalie_label = one_hot_labels2


model = Sequential()
opt = Adam(lr=0.01)
inputs = Input(shape=(None, num_features))
x1 = Bidirectional(GRU(look_back,return_sequences=True), input_shape=(None, num_features) )(inputs)
#model.add(Dropout(0.2))
x1 = Bidirectional(GRU(look_back,return_sequences=True), input_shape=(None, num_features) )(x1)
#model.add(Dropout(0.2))
x1 = Bidirectional(GRU(look_back,return_sequences=True), input_shape=(None, num_features) )(x1)
source_class = (Dropout(0.2))(x1)
source_class = (TimeDistributed(Dense(num_classes,activation = 'tanh')))(source_class)
source_class = (Activation('softmax'))(source_class)



#pd_class = Dense(128, activation='relu')(x1)
#pd_class = Dense(128, activation='relu')(pd_class)
#pd_class = (Dropout(0.2))(x1)
pd_class = TimeDistributed(Dense(1,activation = 'tanh'))(x1)
pd_class = (Activation('relu'))(pd_class)

filepath="weights-improvement3-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint1 = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=True, mode='min')
checkpoint = EarlyStopping(monitor='loss', mode='min', verbose=1, patience = 2)
callbacks_list = [checkpoint,checkpoint1]



comb_model = Model(inputs = [inputs], outputs=[source_class, pd_class])
comb_model.compile(optimizer= "Adam", loss=['categorical_crossentropy','mse'],
              metrics=['categorical_accuracy','mean_absolute_error'])


history = comb_model.fit( [rosalie_input], [rosalie_label, pd_label], epochs=20, batch_size=5,  verbose=2,validation_split = 0.25, callbacks = callbacks_list)



"""
va = model.predict(rosalie_input)
mali=[]
for j in range(len(va)):
    bc = va[j,:]
    bc = bc[look_back-1,:]
    if j == 0:
        mali = bc
    else:
        mali = np.vstack((mali,bc))

print(mali.shape)
workbook = xlsxwriter.Workbook('rosalie_result_oneday.xlsx')
worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(mali.T):
    worksheet.write_column(row, col, data)
workbook.close()
"""


comb_model.save('moneaux13')


