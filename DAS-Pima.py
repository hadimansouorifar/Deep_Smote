from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras import optimizers
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.optimizers import SGD,Adam
import numpy
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import roc_curve
from imblearn.under_sampling import RandomUnderSampler
import random

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import csv
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier

OPTIMIZER = Adam(lr=0.0002, decay=8e-9)

def centroid(arr):
    ind = random.randint(0, len(minority) - 1)
    b = len(arr)
    center = arr[0]
    c = 0
    for i in range(1, b):
        center = center + arr[i]

    center = center / b
    return center

def discriminator():
    model = Sequential()
    model.add(Dense(output_dim=1, input_dim=8, init='uniform'))



    return model


def regression():
    model = Sequential()
    model.add(Dense(64, input_dim=16))
    model.add(Dense(48))
    model.add(Dense(32))
    model.add(Dense(24))
    model.add(Dense(12))
    model.add(Dense(output_dim = 8))

    return model


def stacked():
    disc.trainable = False

    model = Sequential()
    model.add(reg)
    model.add(disc)

    return model



reg= regression()
reg.compile(loss='binary_crossentropy', optimizer=OPTIMIZER)
disc=discriminator()
disc.compile(loss='binary_crossentropy', optimizer=OPTIMIZER)
stack=stacked()
stack.compile(loss='binary_crossentropy', optimizer=OPTIMIZER)



dataset = pd.read_csv("taiwan.csv")
dataset.drop(['ID'], axis=1, inplace=True)
    #print(x.shape)


filename = 'pima.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
td = list(reader)
data = numpy.array(td).astype('str')



x = data[:, 0:8]  # select columns 1 through end

imp = Imputer(missing_values="NaN", strategy='median', axis=0)
x = imp.fit_transform(x)

x = numpy.array(x).astype('float')

w = data[:, 8]   # select column 0, the stock price

y=[]


dim=x.shape
print(dim[0])
c1=0
c2=0
for i in range(0, dim[0]):

    if (w[i]=='1'):
        y.append(1)
        c1=c1+1
    else:
        y.append(0)
        c2=c2+1

ntrain=2000
majority = numpy.zeros((c2,8), dtype=numpy.float)
minority =numpy.zeros((c1,8), dtype=numpy.float)
r1=numpy.zeros((c2-c1,8), dtype=numpy.float)
r2=numpy.zeros((c2-c1,8), dtype=numpy.float)
cent=numpy.zeros((ntrain,8), dtype=numpy.float)

new = numpy.zeros((c1,16), dtype=numpy.float)


r10=numpy.zeros((ntrain,8), dtype=numpy.float)
r20=numpy.zeros((ntrain,8), dtype=numpy.float)
new2 = numpy.zeros((ntrain,16), dtype=numpy.float)

c1=0
c2=0
for i in range(0,len(x)):
        if (y[i]==1):
            minority[c1]=(x[i])
            c1=c1+1
        else:
            majority[c2]=(x[i])
            c2=c2+1


#############################################
for i in range(0,len(majority)-len(minority)):
    ind = random.randint(0, len(minority)-1)
    r1[i]=(minority[ind])

for i in range(0,len(majority)-len(minority)):
    ind = random.randint(0, len(minority)-1)
    r2[i]=(minority[ind])

c=0
for i in range(0,len(majority)-len(minority)):

    new[i]=np.concatenate((r1[i], r2[i]), axis=None)

######################################################3

for i in range(0,len(majority)-len(minority)):
    ind = random.randint(0, len(minority)-1)
    r10[i]=(minority[ind])

for i in range(0,len(majority)-len(minority)):
    ind = random.randint(0, len(minority)-1)
    r20[i]=(minority[ind])

c=0
for i in range(0,len(majority)-len(minority)):

    new2[i]=np.concatenate((r10[i], r20[i]), axis=None)
#############################################################




for k in range(0,len(r10)):
   ce = []
   ce.append(r10[k])
   ce.append(r20[k])
   cent[k]=centroid(ce)

#hist = ae.fit(minority,minority, epochs=150, verbose=0, validation_split=0.2)

def train( X_train, epochs=150, batch=23, save_interval=200):
    gloss = []
    dloss = []
    epoch = []
    maxauc = 0
    d_loss=10
    old=0
    cnt=0
    g_loss=1000
    while ( cnt<200):
        syn=reg.predict(X_train)

        x_combined_batch = np.concatenate((minority, syn))
        y_combined_batch = np.concatenate((np.ones((len(minority), 1)), np.zeros((len(syn), 1))))
        d_loss = disc.train_on_batch(x_combined_batch, y_combined_batch)
        y_mislabled = np.ones((len(X_train), 1))
        g_loss = stack.train_on_batch(X_train, y_mislabled)
        gloss.append(g_loss)
        dloss.append(d_loss)
        cnt=cnt+1
        print('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, dloss[0], g_loss))


        old=g_loss






train(new2)
y_pred=reg.predict(new)

pca = PCA(n_components=2)
pca.fit(y_pred)
sam=pca.transform(y_pred)
pca.fit(minority)
mino=pca.transform(minority)

plt.scatter(mino[:, 0],mino[:, 1],c='red')
plt.scatter(sam[:, 0],sam[:, 1],c='blue',marker='.')
plt.show()


x_combined_mino = np.concatenate((y_pred, minority))
y_combined_mino = np.concatenate((np.ones((len(majority)-len(minority), 1)), np.ones((len(minority), 1))))



x_combined = np.concatenate((x_combined_mino, majority))
y_combined = np.concatenate((np.ones((len(x_combined_mino), 1)), np.zeros((len(majority), 1))))

clf = DecisionTreeClassifier(random_state=0)


sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_sample(x, y)


kf = KFold(n_splits=10,shuffle=True)
c = 0
acc = []
f1 = []
reca=[]
prec=[]

for train, test in kf.split(x_combined):
    train_x = x_combined[train, :]
    test_x = x_combined[test, :]
    train_y = y_combined[train]
    test_y = y_combined[test]

    clf.fit(train_x, train_y)
    predictions = clf.predict(test_x)

    acc.append(accuracy_score(test_y, predictions))
    rec = recall_score(test_y, predictions, labels=None, pos_label=1, average='binary', sample_weight=None)
    reca.append(rec)
    prec.append(precision_score(test_y, predictions, average='micro'))
    f1.append(f1_score(test_y, predictions, average='micro'))
    # score = roc_auc_score(y, round)
    # print(score)


print(np.mean(acc))
print(np.mean(rec))
print(np.mean(prec))
print(np.mean(f1))

####################################################################################


kf = KFold(n_splits=10,shuffle=True)
c = 0
acc = []
f1 = []
reca=[]
prec=[]
for train, test in kf.split(X_train_res):
    train_x = X_train_res[train, :]
    test_x = X_train_res[test, :]
    train_y = y_train_res[train]
    test_y = y_train_res[test]

    clf.fit(train_x, train_y)
    predictions = clf.predict(test_x)

    acc.append(accuracy_score(test_y, predictions))
    rec = recall_score(test_y, predictions, labels=None, pos_label=1, average='binary', sample_weight=None)
    reca.append(rec)
    prec.append(precision_score(test_y, predictions, average='micro'))
    f1.append(f1_score(test_y, predictions, average='micro'))
#    score = roc_auc_score(y, round)
 #   print(score)

print('******************************')
print(np.mean(acc))
print(np.mean(rec))
print(np.mean(prec))
print(np.mean(f1))

