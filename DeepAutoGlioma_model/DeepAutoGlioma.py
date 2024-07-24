#4. Development of DL model (DeepAutoGlioma)
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score,fbeta_score,f1_score,roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
import random
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import random
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

random.seed(10)

random.seed(10)
import matplotlib.pyplot as plt
plt.style.use("ggplot")

df1 =  pd.read_csv("bottledPD_df_400_features.csv")
df = df1
df

# class distribution
# diagnosis: B = 0, M = 1
print(df['y'].value_counts(normalize=True))
targets = df['y']
df.drop([ 'y'], axis=1, inplace=True)

from sklearn import preprocessing
#targets = preprocessing.label_binarize(targets, classes=[1, 2, 3])
targets

X_trainFull=df
y_trainFull=targets

from keras.models import Sequential
#importing keras
import keras
#importing sequential module
from keras.models import Sequential
# import dense module for hidden layers
from keras.layers import Dense
#importing activation functions
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.layers import Dense, Flatten, Conv1D
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

from keras.layers import Dropout
from keras.constraints import maxnorm

    # create model
model = Sequential()
model.add(Conv1D(filters=1, kernel_size=3, activation='relu', input_shape=(400,1)))
model.add(Dropout(0.2))
model.add(Conv1D(filters=1, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(1))
model.add(Flatten())
model.add(Dense(100, activation='softsign'))
#model.add(Dense(50, activation='softsign'))
model.add(Dense(3,activation='softmax'))
    # Compile model
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['categorical_accuracy'])
    

#compiling the ANN
import keras
from keras.callbacks import EarlyStopping

# early stopping callback
# This callback will stop the training when there is no improvement in  
# the validation loss for 10 consecutive epochs.  
es = keras.callbacks.EarlyStopping(monitor='categorical_accuracy', 
                                   mode='max',
                                   patience=200, 
                                   restore_best_weights=True)

from pycm import ConfusionMatrix
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
lb = LabelEncoder()

from sklearn.metrics import matthews_corrcoef
from pycm import ConfusionMatrix

def classification_performance(cnf_matrix,classdist):
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    TPR_WEIGHTED=[a * b for a, b in zip(classdist, TPR)]
    TPR_WEIGHTED=np.sum(TPR_WEIGHTED)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    TNR_WEIGHTED=[a * b for a, b in zip(classdist, TNR)]
    TNR_WEIGHTED=np.sum(TNR_WEIGHTED)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    #PPV_WEIGHTED=[a * b for a, b in zip(classdist, PPV)]
    #print('calculated ',np.sum(PPV_WEIGHTED))
    #print(np.average(PPV, weights=classdist))
    #PPV_WEIGHTED=np.sum(PPV_WEIGHTED)
    
    PPV=[0 if x != x else x for x in PPV]

    PPV_WEIGHTED=[a * b for a, b in zip(classdist, PPV)]
    PPV_WEIGHTED=np.sum(PPV_WEIGHTED)
    
    # Negative predictive value
    NPV = TN/(TN+FN)
    NPV_WEIGHTED=[a * b for a, b in zip(classdist, NPV)]
    NPV_WEIGHTED=np.sum(NPV_WEIGHTED)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    FPR_WEIGHTED = [a * b for a, b in zip(classdist, FPR)]
    FPR_WEIGHTED=np.sum(FPR_WEIGHTED)
    # False negative rate
    FNR = FN/(TP+FN)
    FNR_WEIGHTED=[a * b for a, b in zip(classdist, FNR)]
    FNR_WEIGHTED=np.sum(FNR_WEIGHTED)
    # False discovery rate
    FDR = FP/(TP+FP)
    FDR_WEIGHTED=[a * b for a, b in zip(classdist, FDR)]
    FDR_WEIGHTED=np.sum(FDR_WEIGHTED)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    ACC_WEIGHTED=[a * b for a, b in zip(classdist, ACC)]
    ACC_WEIGHTED=np.sum(ACC_WEIGHTED)
    #f1 score
    F1 = 2*((TPR*PPV)/(TPR+PPV))
    print(F1)
    F1=[0 if x != x else x for x in F1]
    print(F1)
    F1_WEIGHTED=[a * b for a, b in zip(classdist, F1)]
    F1_WEIGHTED=np.sum(F1_WEIGHTED)
    #gmean score
    GM=1
    for a in ACC:
        GM=GM*a
    print(cnf_matrix)
    print('---------------------------------------------------------')
    print('TP',TP)
    print('FP',FP)
    print('FN',FN)
    print('TN',TN)
    print('---------------------------------------------------------')
    GM=np.cbrt(GM)
    print(ACC_WEIGHTED,TPR_WEIGHTED,TNR_WEIGHTED,PPV_WEIGHTED,F1_WEIGHTED,FPR_WEIGHTED,FNR_WEIGHTED,NPV_WEIGHTED,GM)
    
    print('=========================================================')
    
    return ACC_WEIGHTED,TPR_WEIGHTED,TNR_WEIGHTED,PPV_WEIGHTED,F1_WEIGHTED,FPR_WEIGHTED,FNR_WEIGHTED,NPV_WEIGHTED,GM

import statistics as st

folds=10
skf = StratifiedKFold(n_splits=folds)
acc=np.zeros(folds, dtype=object)
recall=np.zeros(folds, dtype=object)
spec=np.zeros(folds, dtype=object)
precision=np.zeros(folds, dtype=object)
F1=np.zeros(folds, dtype=object)
GM=np.zeros(folds, dtype=object)
FPR=np.zeros(folds, dtype=object)
FNR=np.zeros(folds, dtype=object)
NPV=np.zeros(folds, dtype=object)
MCC=np.zeros(folds, dtype=object)

acc_Test=np.zeros(folds, dtype=object)
recall_Test=np.zeros(folds, dtype=object)
spec_Test=np.zeros(folds, dtype=object)
precision_Test=np.zeros(folds, dtype=object)
F1_Test=np.zeros(folds, dtype=object)
GM_Test=np.zeros(folds, dtype=object)
FPR_Test=np.zeros(folds, dtype=object)
FNR_Test=np.zeros(folds, dtype=object)
NPV_Test=np.zeros(folds, dtype=object)
MCC_Test=np.zeros(folds, dtype=object)
iter=0;

for train_index, test_index in skf.split(X_trainFull, y_trainFull):
    

    print('fold start')
    X_train_split, X_test_split = X_trainFull.iloc[train_index], X_trainFull.iloc[test_index]
    y_train_split, y_test_split = y_trainFull.iloc[train_index], y_trainFull.iloc[test_index]
    #y_train=ytrain.iloc[train_index]
    classdist=list(y_train_split.value_counts(normalize=True))
    #print(classdist)
    y_test_split=preprocessing.label_binarize(y_test_split, classes=[1, 2, 3])
    y_train_split=lb.fit_transform(y_train_split)
    dummy_y = np_utils.to_categorical(y_train_split)
    encode_X_train_split=X_train_split.to_numpy().reshape(X_train_split.to_numpy().shape[0],X_train_split.to_numpy().shape[1],1)
    encode_X_test_split=X_test_split.to_numpy().reshape(X_test_split.to_numpy().shape[0],X_test_split.to_numpy().shape[1],1)
    
    history=model.fit(encode_X_train_split, dummy_y,batch_size=64,epochs=2000,shuffle=True,callbacks=[es],verbose=0)
    
    print('TRAIN')
    y_pred_train_split=model.predict(encode_X_train_split)
    MCC[iter]=matthews_corrcoef(dummy_y.argmax(axis=1), y_pred_train_split.argmax(axis=1))
    print('MCC train',MCC[iter])
    #print(y_pred_train_split)
    #print('formuldated ',precision_score(y_train_split, preprocessing.label_binarize(y_pred_train_split,classes=[0,1, 2]), average="macro"))
    conf_matrix=confusion_matrix(dummy_y.argmax(axis=1), y_pred_train_split.argmax(axis=1))
    acc[iter], recall[iter], spec[iter], precision[iter], F1[iter], FPR[iter],FNR[iter],NPV[iter],GM[iter] =classification_performance(conf_matrix,classdist)
    print('TEST')
    y_pred_test_split = model.predict(encode_X_test_split)
    MCC_Test[iter]=matthews_corrcoef(y_test_split.argmax(axis=1), y_pred_test_split.argmax(axis=1))
    print('MCC test',MCC_Test[iter])
    
    conf_matrix=confusion_matrix(y_test_split.argmax(axis=1), y_pred_test_split.argmax(axis=1))
    acc_Test[iter], recall_Test[iter], spec_Test[iter], precision_Test[iter], F1_Test[iter], FPR_Test[iter],FNR_Test[iter],NPV_Test[iter],GM_Test[iter]=classification_performance(conf_matrix,classdist)
    iter=iter+1;
    print('fold end ****************************************************',iter)
    print()

mean_acc=st.mean(acc)
mean_recall=st.mean(recall)
mean_spec=st.mean(spec)
mean_precision=st.mean(precision)
mean_F1=st.mean(F1)
mean_FPR=st.mean(FPR)
mean_FNR=st.mean(FNR)
mean_NPV=st.mean(NPV)
mean_GM=st.mean(GM)
mean_MCC=st.mean(MCC)

print("Mean Train: Accuracy=  %0.6f, Recall=%0.6f, specificity=%0.6f, precision=%0.6f, F1=%0.6f, FPR=%0.6f, FNR=%0.6f, NPV=%0.6f, GM=%0.6f,MCC=%0.6f" % (mean_acc, mean_recall, mean_spec, mean_precision, mean_F1, mean_FPR, mean_FNR, mean_NPV, mean_GM,mean_MCC))

stdDev_acc=st.stdev(acc)
stdDev_recall=st.stdev(recall)
stdDev_spec=st.stdev(spec_Test)
stdDev_precision=st.stdev(precision)
stdDev_F1=st.stdev(F1)
stdDev_FPR=st.stdev(FPR)
stdDev_FNR=st.stdev(FNR)
stdDev_NPV=st.stdev(NPV)
stdDev_GM=st.stdev(GM)
stdDev_MCC=st.stdev(MCC)

print("std Train: Accuracy= %0.6f, Recall=%0.6f, specificity=%0.6f, precision=%0.6f, F1=%0.6f, FPR=%0.6f, FNR=%0.6f, NPV=%0.6f, GM=%0.6f ,MCC=%0.6f" % (stdDev_acc, stdDev_recall, stdDev_spec, stdDev_precision, stdDev_F1, stdDev_FPR, stdDev_FNR, stdDev_NPV, stdDev_GM, stdDev_MCC))



mean_acc=st.mean(acc_Test)
mean_recall=st.mean(recall_Test)
mean_spec=st.mean(spec_Test)
mean_precision=st.mean(precision_Test)
mean_F1=st.mean(F1_Test)
mean_FPR=st.mean(FPR_Test)
mean_FNR=st.mean(FNR_Test)
mean_NPV=st.mean(NPV_Test)
mean_GM=st.mean(GM_Test)
mean_MCC=st.mean(MCC_Test)
print("Mean TEST:  Accuracy=  %0.6f, Recall=%0.6f, specificity=%0.6f, precision=%0.6f, F1=%0.6f, FPR=%0.6f, FNR=%0.6f, NPV=%0.6f, GM=%0.6f,MCC=%0.6f" % (mean_acc, mean_recall, mean_spec, mean_precision, mean_F1, mean_FPR, mean_FNR, mean_NPV, mean_GM,mean_MCC))


stdDev_acc=st.stdev(acc_Test)
stdDev_recall=st.stdev(recall_Test)
stdDev_spec=st.stdev(spec_Test)
stdDev_precision=st.stdev(precision_Test)
stdDev_F1=st.stdev(F1_Test)
stdDev_FPR=st.stdev(FPR_Test)
stdDev_FNR=st.stdev(FNR_Test)
stdDev_NPV=st.stdev(NPV_Test)
stdDev_GM=st.stdev(GM_Test)
stdDev_MCC=st.stdev(MCC_Test)
print("std Test: Accuracy=  %0.6f, Recall=%0.6f, specificity=%0.6f, precision=%0.6f, F1=%0.6f, FPR=%0.6f, FNR=%0.6f, NPV=%0.6f, GM=%0.6f,MCC=%0.6f" % (stdDev_acc, stdDev_recall, stdDev_spec, stdDev_precision, stdDev_F1, stdDev_FPR, stdDev_FNR, stdDev_NPV, stdDev_GM,stdDev_MCC ))


encode_X_train=X_trainFull.to_numpy().reshape(X_trainFull.to_numpy().shape[0],X_trainFull.to_numpy().shape[1],1)
dummy_y = np_utils.to_categorical(lb.fit_transform(y_trainFull))
model.fit(encode_X_train, dummy_y,batch_size=50,epochs=2000,shuffle=True,callbacks=[es],verbose=1)   
encode_X_test=X_test.to_numpy().reshape(X_test.to_numpy().shape[0],X_test.to_numpy().shape[1],1)
    
#clf.fit(X_train, y_train.argmax(axis=1))
y_pred1 = model.predict(encode_X_test)

conf_matrix=confusion_matrix(y_test_labelBinarized.argmax(axis=1), y_pred1.argmax(axis=1))
print(conf_matrix) 

MCC_FINAL_TEST = matthews_corrcoef(y_test_labelBinarized.argmax(axis=1), y_pred1.argmax(axis=1))
print('MCC_FINAL_TEST',MCC_FINAL_TEST)

classdist=list(y_test.value_counts(normalize=True))
acc_Test, recall_Test, spec_Test, precision_Test, F1_Test, FPR_Test,FNR_Test,NPV_Test,GM_Test= classification_performance(conf_matrix,classdist)
print(" TEST:  Accuracy= %0.4f, Recall=%0.4f, specificity=%0.4f, precision=%0.4f, F1=%0.4f, FPR=%0.4f, FNR=%0.4f, NPV=%0.4f, GM=%0.4f" % (acc_Test, recall_Test, spec_Test, precision_Test,F1_Test, FPR_Test, FNR_Test,NPV_Test, GM_Test))

cnf_matrix = conf_matrix
print(cnf_matrix)
#[[1 1 3]
# [3 2 2]
# [1 3 1]]

FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
#f1 score
F1 = 2*((TPR*PPV)/(TPR+PPV))


print('Overall accuracy')
print('Overall accuracy: {:.2f}'.format,ACC)

print('Sensitivity')
print('Weighted Sensitivity: {:.2f}'.format,TPR)

print('Specificity')
print('Weighted Specificity: {:.2f}'.format, TNR)

print('Precision')
print('Weighted Precision: {:.2f}'.format, PPV)

print('NPV')
print('Weighted Negative predictive value: {:.2f}'.format, NPV)

print('FPR')
print('Weighted false positive rate: {:.2f}'.format,FPR)

print('FNR')
print('Weighted False negative rate: {:.2f}'.format,FNR)

print('FDR')
print('Weighted False discovery rate: {:.2f}'.format,FDR)

print('F1 score')
print('Weighted False discovery rate: {:.2f}'.format,F1)

from scipy.stats.mstats import gmean
gmean_list=gmean([TPR,TNR])
print('gmean')
print('Weighted False discovery rate: {:.2f}'.format,gmean_list)

n_classes=3
noOfSamplesSelected='100'

# roc curve
import seaborn as sns
sns.set(font_scale=3, style='white')
fpr = dict()
tpr = dict()


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.metrics import roc_auc_score
y_test=y_test_labelBinarized
y_score=y_pred1
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i],
                                  y_score[:, i])
    plt.plot(fpr[i], tpr[i], lw=2, label='class {}'.format(i))

plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.legend(loc="best")
plt.title("ROC curve")
plt.show()


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
fig = plt.figure(figsize=(17, 12), dpi=300)
#plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([-0.05, 1], [-0.05, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()
#fileSaveNam001="G2_KNN_single"+noOfSamplesSelected+".tif"

#fig.savefig(fileSaveNam001,  bbox_inches='tight',dpi=300)

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["weighted"] = all_fpr
tpr["weighted"] = mean_tpr
roc_auc["weighted"] = auc(fpr["weighted"], tpr["weighted"])

# Plot all ROC curves
fig = plt.figure(figsize=(17, 12), dpi=300)
#plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

fig = plt.figure(figsize=(17, 12), dpi=300)
plt.plot(fpr["weighted"], tpr["weighted"],
         label='(AUC = {0:0.2f})'
               ''.format(roc_auc["weighted"]),
         color='navy', linestyle=':', linewidth=4)
plt.plot([-0.05, 1], [-0.05, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('1 - Specificity', fontsize=24)
plt.ylabel('Sensitivity', fontsize=24)
#plt.title('ROC')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc="lower right")
plt.show()
fileSaveNam01="GBM_Integrated_weighted_final_CNN"+noOfSamplesSelected+".tif"

fig.savefig(fileSaveNam01,  bbox_inches='tight',dpi=300)


fig = plt.figure(figsize=(15, 10), dpi=300)


plt.plot(fpr["weighted"], tpr["weighted"],
         label='ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["weighted"]),
             color='cornflowerblue', linewidth=2)

colors = cycle(['red', 'green', 'blue'])
class_names = cycle(['Classical', 'Mesenchymal', 'Proneural'])
for i, color,className in zip(range(n_classes), colors,class_names):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='{0} (area = {1:0.2f})'
             ''.format(className, roc_auc[i]))

plt.plot([-0.05, 1], [-0.05, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('1 - Specificity', fontsize=40)
plt.ylabel('Sensitivity', fontsize=40)
#plt.title('ROC')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

#plt.title('ROC for all class')
plt.legend(loc="lower right")
plt.show()
fileSaveNam1="GBM_Integrated_final_CNN"+noOfSamplesSelected+".tif"

fig.savefig(fileSaveNam1,  bbox_inches='tight',dpi=300)
