from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras import optimizers
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.optimizers import SGD,Adam
import numpy
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_curve
from imblearn.under_sampling import RandomUnderSampler
import random
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.tree import DecisionTreeClassifier

import csv
OPTIMIZER = Adam(lr=0.0002, decay=8e-9)







import pandas
url = "haberman.csv"
names = ['Age', 'Year operation', 'Axillary nodes detected', 'Survival status']
dataset = pandas.read_csv(url, names=names)
dataset.head(5)
array = dataset.values
X = array[:,:3]
Y = array[:,3]

x=X

w=Y
dim=x.shape
print(dim[0])
c1=0
c2=0
y=[]
for i in range(0, dim[0]):

    if (w[i]==2):
        w[i]=1
        c1=c1+1
    else:
        w[i]=0
        c2=c2+1

amount_scaler = StandardScaler().fit(x[:])
x[:] = amount_scaler.transform(x[:])



y=[]



ntrain=750
def discriminator():
    model= Sequential()
    model.add(Dense(10, input_dim=12, init='uniform'))


    model.add(Dense(8))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    return model


def regression():
    model = Sequential()
    model.add(Dense(36, input_dim=12))
    model.add(Dense(24))

    model.add(Dense(output_dim = 12))

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

for mm in range(0,3):
#    kf = KFold(n_splits=10, shuffle=True)
    kf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
    c = 0

    acc = []
    f1 = []
    reca = []
    prec = []
    auc = []
    ##################3
    accsvm = []
    f1svm = []
    recasvm = []
    precsvm = []
    aucsvm = []
    #####################
    accnaive = []
    f1naive = []
    recanaive = []
    precnaive = []
    aucnaive = []
    #####################
    acc2 = []
    f12 = []
    reca2 = []
    prec2 = []
    auc2 = []
    ##################3
    acc2svm = []
    f12svm = []
    reca2svm = []
    prec2svm = []
    auc2svm = []
    #####################
    acc2naive = []
    f12naive = []
    reca2naive = []
    prec2naive = []
    auc2naive = []
    #####################
    for train, test in kf.split(x,w):

        train_x = x[train, :]
        test_x = x[test, :]
        train_y = w[train]
        test_y = w[test]

        dim = train_x.shape
        print(dim[0])
        c1 = 0
        c2 = 0
        for i in range(0, dim[0]):

            if (train_y[i] == 1):
                # y.append(1)
                c1 = c1 + 1
            else:
                # y.append(0)
                c2 = c2 + 1

        majority = numpy.zeros((c2, 3), dtype=numpy.float)
        minority = numpy.zeros((c1, 3), dtype=numpy.float)
        r1 = numpy.zeros((c2 - c1, 3), dtype=numpy.float)
        r2 = numpy.zeros((c2 - c1, 3), dtype=numpy.float)
        cent = numpy.zeros((ntrain, 3), dtype=numpy.float)


        r10 = numpy.zeros((ntrain, 3), dtype=numpy.float)
        r20 = numpy.zeros((ntrain, 3), dtype=numpy.float)
        new2 = numpy.zeros((ntrain, 12), dtype=numpy.float)

        c1 = 0
        c2 = 0
        for i in range(0, len(train_x)):
            if (train_y[i] == 1):
                minority[c1] = (train_x[i])
                c1 = c1 + 1
            else:
                majority[c2] = (train_x[i])
                c2 = c2 + 1

        for i in range(0, ntrain):
            ind1 = random.randint(0, len(minority) - 1)
            ind2 = random.randint(0, len(minority) - 1)
            ind3 = random.randint(0, len(minority) - 1)
            ind4 = random.randint(0, len(minority) - 1)

            t1 = np.concatenate((minority[ind1], minority[ind2]), axis=None)
            t2 = np.concatenate((minority[ind3], minority[ind4]), axis=None)
            new2[i] = np.concatenate((t1, t2), axis=None)

        n1 = 0
        n2 = 1


        def train(X_train, epochs=30, batch=64, save_interval=200):
            gloss = []
            dloss = []
            epoch = []
            for cnt in range(epochs):
                ## train discriminator
                legit_images = X_train
                gen_noise = np.random.uniform(n1, n2, size=[len(X_train), 12])
                syntetic_images = reg.predict(gen_noise)
                print(syntetic_images.shape)
                # print(legit_images.)

                x_combined_batch = np.concatenate((legit_images, syntetic_images))
                y_combined_batch = np.concatenate(
                    (np.ones((len(legit_images), 1)), np.zeros((len(syntetic_images), 1))))
                d_loss = disc.train_on_batch(x_combined_batch, y_combined_batch)
                # train generator
                noise = np.random.normal(n1, n2, (10, 12))
                y_mislabled = np.ones((10, 1))
                g_loss = stack.train_on_batch(noise, y_mislabled)
                gloss.append(g_loss)
                dloss.append(d_loss)
                epoch.append(cnt)
                print('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, dloss[0], g_loss))


        train(new2)

        noise = np.random.normal(n1, n2, (len(majority) - len(minority), 12))

        y_pred = reg.predict(noise)

        print(y_pred[0])
        print(minority[0])

        fin = []
        for i in range(0, len(y_pred)):
            fin.append(y_pred[i][0:3])

        print(fin[0])

        pca = PCA(n_components=2)
        pca.fit(fin)
        sam = pca.transform(fin)
        pca.fit(minority)
        mino = pca.transform(minority)

        plt.scatter(mino[:, 0], mino[:, 1], c='red')
        plt.scatter(sam[:, 0], sam[:, 1], c='blue', marker='.')
        # plt.show()

        temp = np.concatenate((fin, minority))
        x_combined = np.concatenate((temp, majority))
        y_combined = np.concatenate((np.ones((len(majority), 1)), np.zeros((len(majority), 1))))

        clf = DecisionTreeClassifier(random_state=0)

        clf.fit(x_combined, y_combined)
        predictions = clf.predict(test_x)

        tt1 = numpy.array(test_y).astype('int')
        tt2 = numpy.array(predictions).astype('int')
        test_y=numpy.array(test_y).astype('int')
        predictions=numpy.array(predictions).astype('int')

        auct = roc_auc_score(tt1, tt2)

        acc.append(accuracy_score(test_y, predictions))
        rec = recall_score(test_y, predictions, labels=None, pos_label=1, average='binary', sample_weight=None)
        reca.append(rec)
        prec.append(precision_score(test_y, predictions, pos_label=1, average='binary'))
        f1.append(f1_score(test_y, predictions, pos_label=1, average='binary'))
        auc.append(auct)
        # score = roc_auc_score(y, round)
        # print(score)

        ###################################################
        clf = svm.SVC(gamma=0.001)
        clf.fit(x_combined, y_combined)
        predictions = clf.predict(test_x)

        tt1 = numpy.array(test_y).astype('int')
        tt2 = numpy.array(predictions).astype('int')
        test_y = numpy.array(test_y).astype('int')
        predictions=numpy.array(predictions).astype('int')

        auct = roc_auc_score(tt1, tt2)

        accsvm.append(accuracy_score(test_y, predictions))
        rec = recall_score(test_y, predictions, labels=None, pos_label=1, average='binary', sample_weight=None)
        recasvm.append(rec)
        precsvm.append(precision_score(test_y, predictions, pos_label=1, average='binary'))
        f1svm.append(f1_score(test_y, predictions, pos_label=1, average='binary'))
        aucsvm.append(auct)
        ###########################################################################

        clf = GaussianNB()
        clf.fit(x_combined, y_combined)
        predictions = clf.predict(test_x)

        tt1 = numpy.array(test_y).astype('int')
        tt2 = numpy.array(predictions).astype('int')
        test_y = numpy.array(test_y).astype('int')
        predictions=numpy.array(predictions).astype('int')

        auct = roc_auc_score(tt1, tt2)

        accnaive.append(accuracy_score(test_y, predictions))
        rec = recall_score(test_y, predictions, labels=None, pos_label=1, average='binary', sample_weight=None)
        recanaive.append(rec)
        precnaive.append(precision_score(test_y, predictions, pos_label=1, average='binary'))
        f1naive.append(f1_score(test_y, predictions, pos_label=1, average='binary'))
        aucnaive.append(auct)
        ###########################################################################


        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_sample(train_x, train_y)

        clf = DecisionTreeClassifier(random_state=0)

        clf.fit(X_train_res, y_train_res)
        predictions = clf.predict(test_x)
        tt1 = numpy.array(test_y).astype('float')
        tt2 = numpy.array(predictions).astype('float')
        test_y = numpy.array(test_y).astype('int')
        predictions=numpy.array(predictions).astype('int')
        auct = roc_auc_score(tt1, tt2)

        print(predictions)

        acc2.append(accuracy_score(test_y, predictions))
        rec2 = recall_score(test_y, predictions, labels=None, pos_label=1, average='binary', sample_weight=None)
        reca2.append(rec2)
        prec2.append(precision_score(test_y, predictions, pos_label=1, average='binary'))
        f12.append(f1_score(test_y, predictions, pos_label=1, average='binary'))
        auc2.append(auct)
        ###################################################
        clf = svm.SVC(gamma=0.001)
        clf.fit(X_train_res, y_train_res)
        predictions = clf.predict(test_x)

        tt1 = numpy.array(test_y).astype('int')
        tt2 = numpy.array(predictions).astype('int')
        test_y = numpy.array(test_y).astype('int')
        predictions=numpy.array(predictions).astype('int')

        auct = roc_auc_score(tt1, tt2)

        acc2svm.append(accuracy_score(test_y, predictions))
        rec = recall_score(test_y, predictions, labels=None, pos_label=1, average='binary', sample_weight=None)
        reca2svm.append(rec)
        prec2svm.append(precision_score(test_y, predictions, pos_label=1, average='binary'))
        f12svm.append(f1_score(test_y, predictions, pos_label=1, average='binary'))

        auc2svm.append(auct)
        ###########################################################################

        clf = GaussianNB()
        clf.fit(X_train_res, y_train_res)
        predictions = clf.predict(test_x)

        tt1 = numpy.array(test_y).astype('int')
        tt2 = numpy.array(predictions).astype('int')
        test_y = numpy.array(test_y).astype('int')
        predictions=numpy.array(predictions).astype('int')

        auct = roc_auc_score(tt1, tt2)

        acc2naive.append(accuracy_score(test_y, predictions))
        rec = recall_score(test_y, predictions, labels=None, pos_label=1, average='binary', sample_weight=None)
        reca2naive.append(rec)
        prec2naive.append(precision_score(test_y, predictions, pos_label=1, average='binary'))
        f12naive.append(f1_score(test_y, predictions, pos_label=1, average='binary'))
        auc2naive.append(auct)
        ###########################################################################
print("DT Results")
print(np.mean(acc))
print(np.mean(reca))
print(np.mean(prec))
print(np.mean(f1))
print(np.mean(auc))
print('####################')

print(np.mean(acc2))
print(np.mean(reca2))
print(np.mean(prec2))
print(np.mean(f12))
print(np.mean(auc2))

print('#################### STD Deep ********')
print(np.std(acc))
print(np.std(reca))
print(np.std(prec))
print(np.std(f1))
print(np.std(auc))


print('#################### STD SMOTE ********')

print(np.std(acc2))
print(np.std(reca2))
print(np.std(prec2))
print(np.std(f12))
print(np.std(auc2))

    ####################################################################################

print('##############################')
print("SVM Results")
print(np.mean(accsvm))
print(np.mean(recasvm))
print(np.mean(precsvm))
print(np.mean(f1svm))
print(np.mean(aucsvm))
print('####################')

print(np.mean(acc2svm))
print(np.mean(reca2svm))
print(np.mean(prec2svm))
print(np.mean(f12svm))
print(np.mean(auc2svm))
print('##########################################')

print('#################### STD Deep ********')
print(np.std(accsvm))
print(np.std(recasvm))
print(np.std(precsvm))
print(np.std(f1svm))
print(np.std(aucsvm))

print('#################### STD SMOTE ********')

print(np.std(acc2svm))
print(np.std(reca2svm))
print(np.std(prec2svm))
print(np.std(f12svm))
print(np.std(auc2svm))

print('###########################')
print("NB Results")
print(np.mean(accnaive))
print(np.mean(recanaive))
print(np.mean(precnaive))
print(np.mean(f1naive))
print(np.mean(aucnaive))
print('####################')

print(np.mean(acc2naive))
print(np.mean(reca2naive))
print(np.mean(prec2naive))
print(np.mean(f12naive))
print(np.mean(auc2naive))

print('#####################')

print('#################### STD Deep ********')
print(np.std(accnaive))
print(np.std(recanaive))
print(np.std(precnaive))
print(np.std(f1naive))
print(np.std(aucnaive))

print('#################### STD SMOTE ********')

print(np.std(acc2naive))
print(np.std(reca2naive))
print(np.std(prec2naive))
print(np.std(f12naive))
print(np.std(auc2naive))

