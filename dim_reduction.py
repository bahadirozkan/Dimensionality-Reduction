# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#Load the data with the column names
names = ["word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
"word_freq_our", "word_freq_over","word_freq_remove","word_freq_internet",
"word_freq_order","word_freq_mail","word_freq_receive","word_freq_will",
"word_freq_people","word_freq_report","word_freq_addresses","word_freq_free",
"word_freq_business","word_freq_email","word_freq_you","word_freq_credit",
"word_freq_your","word_freq_font","word_freq_000","word_freq_money","word_freq_hp",
"word_freq_hpl","word_freq_george","word_freq_650","word_freq_lab","word_freq_labs",
"word_freq_telnet","word_freq_857","word_freq_data","word_freq_415","word_freq_85",
"word_freq_technology","word_freq_1999","word_freq_parts","word_freq_pm",
"word_freq_direct","word_freq_cs","word_freq_meeting","word_freq_original",
"word_freq_project","word_freq_re","word_freq_edu","word_freq_table","word_freq_conference",
"char_freq_;","char_freq_(","char_freq_[","char_freq_!","char_freq_$","char_freq_#",
"capital_run_length_average","capital_run_length_longest", "capital_run_length_total",
"target"]
data = pd.read_csv('dataset.txt', delimiter = ",", names=names)

data.head()

train = data.iloc[:,:57].values
test = data["target"].values
X_train, X_test, y_train, y_test = train_test_split(
    train, test, test_size=0.5, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#Case 1: Feed the original dataset without any dimensionality reduction as input to k-NN.
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

#train performance
y_pred_tr = model.predict(X_train)
print("Case 1 Train accuracy:", accuracy_score(y_train, y_pred_tr))

#test performance
y_pred_ts = model.predict(X_test)
print("Case 1 Test accuracy:", accuracy_score(y_test, y_pred_ts))

#Precision and recall results for train and test
print("\nCase 1 Precision and Recall for train:\n", classification_report(y_train, y_pred_tr))
print("\nCase 1 Precision and Recall for test:\n", classification_report(y_test, y_pred_ts))

pca = PCA(n_components=57)
pca.fit_transform(X_train)

variance_exp_cumsum = pca.explained_variance_ratio_.cumsum()
plt.plot(variance_exp_cumsum, color='firebrick')
plt.title('Variance Explained %', fontsize=15)
plt.xlabel('# of PCs', fontsize=12)
plt.show()

"""
Case 2: Feature extraction:  Plot the data for m=2.
"""
pca = PCA(n_components=2)
X_trainPCA = pca.fit_transform(X_train)
X_testPCA = pca.fit_transform(X_test)

model.fit(X_trainPCA, y_train)

#train performance
y_pred_tr = model.predict(X_trainPCA)
print("Case 2 Train accuracy:", model.score(X_trainPCA,y_train))

#test performance
y_pred_ts = model.predict(X_testPCA)
print("Case 2 Test accuracy:", model.score(X_testPCA,y_test))

#Precision and recall results for train and test
print("\nCase 2 Precision and Recall for train m=2:\n", classification_report(y_train, y_pred_tr))
print("\nCase 2 Precision and Recall for test m=2:\n", classification_report(y_test, y_pred_ts))

plt.figure(figsize = (6,6))
plt.scatter(X_trainPCA[:,0],X_trainPCA[:,1], c = y_pred_tr, cmap = "brg")
plt.title("PCA Plot for m=2")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

"""
Use PCA to reduce dimensionality to m, followed by k-NN.
Try for different values of m corresponding to proportion
of variance of 0.80, 0.81, 0.82, ...., 0.99.
"""
var_index = range(np.where(variance_exp_cumsum>=0.80)[0][0],len(variance_exp_cumsum)+1)
#empty arrays to store accuracy, precision and recall values for train and test
train_accPCA = []
test_accPCA = []
trainPrecision_PCA = []
testPrecision_PCA = []
trainRecall_PCA = []
testRecall_PCA = []

for i in var_index:
  pca = PCA(n_components=i)
  X_trainPCA = pca.fit_transform(X_train)
  X_testPCA = pca.fit_transform(X_test)

  model.fit(X_trainPCA, y_train)

  #train performance
  y_pred_tr = model.predict(X_trainPCA)
  train_accPCA.append(accuracy_score(y_train,y_pred_tr))

  #test performance
  y_pred_ts = model.predict(X_testPCA)
  test_accPCA.append(accuracy_score(y_test,y_pred_ts))

  #precision and recall values recorded for all k (feature number)
  trainPrecision_PCA.append(precision_score(y_train, y_pred_tr))
  testPrecision_PCA.append(precision_score(y_test, y_pred_ts))

  trainRecall_PCA.append(recall_score(y_train, y_pred_tr))
  testRecall_PCA.append(recall_score(y_test, y_pred_ts))

#m value that gives the highest classification accuracy on the training set (Case 2)
print("m-value on train set that gives the best accuracy:",var_index[np.argmax(train_accPCA)])
ind = np.argmax(train_accPCA)
print("Accuracy, precision and recall values for m-best (Case2/Train):",train_accPCA[ind], trainPrecision_PCA[ind],trainRecall_PCA[ind])

print("Accuracy, precision and recall values for m-best (Case2/Test):",test_accPCA[ind], testPrecision_PCA[ind],testRecall_PCA[ind])

plt.plot(var_index,train_accPCA, label='Accuracy')
plt.plot(var_index,trainPrecision_PCA, label='Precision')
plt.plot(var_index, trainRecall_PCA, label='Recall')
plt.title('Accuracy, Recall, Precision for PCA Train', fontsize=15)
plt.xlabel('# of features', fontsize=12)
plt.legend()
plt.show()

plt.plot(var_index,test_accPCA, label='Accuracy')
plt.plot(var_index,testPrecision_PCA, label='Precision')
plt.plot(var_index, testRecall_PCA, label='Recall')
plt.title('Accuracy, Recall, Precision for PCA Test', fontsize=15)
plt.xlabel('# of features', fontsize=12)
plt.legend()
plt.show()

"""
Feature Selection: plot the data for m=2.
"""
X_trainFS = SelectKBest(chi2,k=2).fit_transform(X_train,y_train)
X_testFS = SelectKBest(chi2,k=2).fit_transform(X_test,y_test)

model.fit(X_trainFS, y_train)

#train performance
y_pred_tr = model.predict(X_trainFS)

#train performance
y_pred_tr = model.predict(X_trainFS)
print("Case 3 Train accuracy m=2:", model.score(X_trainFS,y_train))

#test performance
y_pred_ts = model.predict(X_testFS)
print("Case 3 Test accuracy m=2:", model.score(X_testFS,y_test))

#Precision and recall results for train and test
print("\nCase 3 Precision and Recall for train m=2:\n", classification_report(y_train, y_pred_tr))
print("\nCase 3 Precision and Recall for test m=2:\n", classification_report(y_test, y_pred_ts))

plt.figure(figsize = (6,6))
plt.scatter(X_trainFS[:,0],X_trainFS[:,1], c = y_pred_tr, cmap = "brg")
plt.title("FS Plot for m=2")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

"""
Use forward selection to reduce dimensionality to m using k-NN as predictor.
Train the model for each m between 1 and 57.
"""
#empty arrays to store accuracy, precision and recall values for train and test
train_accFS = []
test_accFS = []
trainPrecision_FS = []
testPrecision_FS = []
trainRecall_FS = []
testRecall_FS = []

for i in range(1,58):
  X_trainFS = SelectKBest(chi2,k=i).fit_transform(X_train,y_train)
  X_testFS = SelectKBest(chi2,k=i).fit_transform(X_test,y_test)

  model.fit(X_trainFS, y_train)

  #train performance
  y_pred_tr = model.predict(X_trainFS)
  train_accFS.append(accuracy_score(y_train,y_pred_tr))

  #test performance
  y_pred_ts = model.predict(X_testFS)
  test_accFS.append(accuracy_score(y_test,y_pred_ts))

  #precision and recall values recorded for all k (feature number)
  trainPrecision_FS.append(precision_score(y_train, y_pred_tr))
  testPrecision_FS.append(precision_score(y_test, y_pred_ts))

  trainRecall_FS.append(recall_score(y_train, y_pred_tr))
  testRecall_FS.append(recall_score(y_test, y_pred_ts))

#m value that gives the highest classification accuracy on the training set (Case 3)
f = range(1,58)
print("m-value on train set that gives the best accuracy:",f[np.argmax(train_accFS)])

#report the classification accuracy, precision, and recall for
ind = np.argmax(train_accFS)
print("Accuracy, precision and recall values for m-best (Case3/Train):",train_accFS[ind], trainPrecision_FS[ind],trainRecall_FS[ind])

print("Accuracy, precision and recall values for m-best (Case3/Test):",test_accFS[ind], testPrecision_FS[ind],testRecall_FS[ind])

plt.plot(f,train_accFS, label='Accuracy')
plt.plot(f,trainPrecision_FS, label='Precision')
plt.plot(f, trainRecall_FS, label='Recall')
plt.title('Accuracy, Recall, Precision for Forward Selection (Train)', fontsize=15)
plt.xlabel('# of features', fontsize=12)
plt.legend()
plt.show()

plt.plot(f,test_accFS, label='Accuracy')
plt.plot(f,testPrecision_FS, label='Precision')
plt.plot(f, testRecall_FS, label='Recall')
plt.title('Accuracy, Recall, Precision for Forward Selection (Test)', fontsize=15)
plt.xlabel('# of features', fontsize=12)
plt.legend()
plt.show()
