import matplotlib.pyplot as plt
import pickle
import time
import math
import numpy as np
import numpy.random as rnd

with open('mnistTVT.pickle','rb') as f:
    Xtrain,Ttrain,Xval,Tval,Xtest,Ttest = pickle.load(f)
Xtrain = Xtrain.astype(np.float64)
Xval = Xval.astype(np.float64)
Xtest = Xtest.astype(np.float64)

Xsmall = Xtrain[:200]
Tsmall = Ttrain[:200]
Xdebug = Xtrain[:300]
Tdebug = Ttrain[:300]

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

print('-------------')
clf = QuadraticDiscriminantAnalysis()
clf.fit(Xsmall, Tsmall)
accuracy1 = clf.score(Xsmall, Tsmall) # train Accuracy
accuracy2 = clf.score(Xtest, Ttest) # test Accuracy
print("\nTrain Acc: %.4f Test Acc: %.4f" % (accuracy1, accuracy2))

print('-------------')

all_train = []
all_val = []
all_reg_param = []
best_n = 0
for n in range(0,20):
  clf2 = QuadraticDiscriminantAnalysis(reg_param=2**(-n))
  clf2.fit(Xsmall, Tsmall)
  accuracy1 = clf2.score(Xsmall, Tsmall) # train Accuracy
  accuracy2 = clf2.score(Xval, Tval) # validation Accuracy
  all_train.append(accuracy1)
  all_val.append(accuracy2)
  all_reg_param.append(2**(-n))
  if accuracy2 > all_val[best_n]:
    best_n = n

print("\nBest regularization parameter: %f \nTrain Acc: %.4f Validation Acc: %.4f" % (all_reg_param[best_n], all_train[best_n], all_val[best_n]))

plt.semilogx( all_reg_param, all_train, color="blue")
plt.semilogx( all_reg_param, all_val, color="red")
plt.title("Training and Validation Accuracy for Regularized QDA")
plt.xlabel("Regularization parameter")
plt.ylabel("Accuracy")
plt.show()

from sklearn.decomposition import PCA

def train2d(K,X,T):
  pca = PCA(n_components=K, svd_solver='full')
  pca.fit(X)
  Xreduced = pca.transform(X)
  qda = QuadraticDiscriminantAnalysis()
  qda.fit(Xreduced, T)
  accuracy = qda.score(Xreduced, T)
  return pca, qda, accuracy

def test2d(pca,qda,X,T):
  Xreduced = pca.transform(X)
  accuracy = qda.score(Xreduced, T)
  return accuracy

print('-------------')
all_train = []
all_val = []
all_k = [] # Note k starts at index 1
best_k = 1
for k in range(1,51):
  pca, qda, train_acc = train2d(k, Xsmall, Tsmall)
  test_acc = test2d(pca, qda, Xval, Tval)
  all_train.append(train_acc)
  all_val.append(test_acc)
  all_k.append(k)
  if test_acc > all_val[best_k-1]:
    best_k = k

print("\nBest K: %f \nTrain Acc: %.4f Validation Acc: %.4f" % (all_k[best_k-1], all_train[best_k-1], all_val[best_k-1]))

plt.plot( all_k, all_train, color="blue")
plt.plot( all_k, all_val, color="red")
plt.title("Training and Validation Accuracy for PCA + QDA")
plt.xlabel("Reduced dimension")
plt.ylabel("Accuracy")
plt.show()

print('-------------')

allbest_train = []
allbest_val = []
all_k = [] # Note k starts at index 1
best_k = 1
best_reg_param = 0
all_accMaxK = []

for k in range(1,51):
  pca = PCA(n_components=k, svd_solver='full')
  pca.fit(Xsmall)
  Xsmall_reduced = pca.transform(Xsmall)
  Xval_reduced = pca.transform(Xval)
  all_train = []
  all_val = []
  all_reg_param = []
  best_n = 0

  for n in range(0,20):
    qda = QuadraticDiscriminantAnalysis(reg_param=2**(-n))
    qda.fit(Xsmall_reduced, Tsmall)
    train_acc = qda.score(Xsmall_reduced, Tsmall)
    val_acc = qda.score(Xval_reduced, Tval)
    all_train.append(train_acc)
    all_val.append(val_acc)
    all_reg_param.append(2**(-n))
    if val_acc > all_val[best_n]:
      best_n = n

  accMaxK = all_val[best_n]
  all_accMaxK.append(accMaxK)
  allbest_train.append(all_train[best_n])
  allbest_val.append(accMaxK)
  accMax = allbest_val[best_k-1]
  all_k.append(k)
  if accMaxK > accMax:
    best_k = k
    best_reg_param = all_reg_param[best_n]
  
print("\nBest K: %f Best reg_param: %f \nTrain Acc: %.4f Validation Acc: %.4f" % (all_k[best_k-1], best_reg_param, allbest_train[best_k-1], accMax))

plt.plot( all_k, all_accMaxK)
plt.title("Maximum validation accuracy for QDA")
plt.xlabel("Reduced dimension")
plt.ylabel("maximum accuracy")
plt.show()

print('-------------')

from sklearn.utils import resample
import random

def myBootstrap(X,T):
  while True:
    Xbootstrap, Tbootstrap = resample(X, T, replace=True)
    count = np.unique(Tbootstrap, return_counts=True)
    if all( 3 <= count[1] ):
      return Xbootstrap, Tbootstrap

print('-------------')
qda = QuadraticDiscriminantAnalysis(reg_param=0.004)
qda.fit(Xsmall, Tsmall)
base_val_acc = qda.score(Xval, Tval)
print("\nBase Validation Acc: %.4f" % (base_val_acc))
averaged_prob_matrix = np.zeros((Xval.shape[0], 10))

for i in range(0, 50): 
  Xbs, Tbs = myBootstrap(Xsmall, Tsmall)
  qda = QuadraticDiscriminantAnalysis(reg_param=0.004)
  qda.fit(Xbs, Tbs)
  prob_matrix = qda.predict_proba(Xval) 
  averaged_prob_matrix = averaged_prob_matrix + prob_matrix

averaged_prob_matrix = averaged_prob_matrix/50
predicted_class = np.argmax(averaged_prob_matrix, axis=1)
bagged_val_acc = 1-(np.count_nonzero(predicted_class - Tval)/Xval.shape[0])
print("\nBagged Validation Acc: %.4f" % (bagged_val_acc))


all_val_acc = []       
all_i = []
averaged_prob_matrix = np.zeros((Xval.shape[0], 10))

for i in range(0, 500): 
  all_i.append(i+1)
  Xbs, Tbs = myBootstrap(Xsmall, Tsmall)
  qda = QuadraticDiscriminantAnalysis(reg_param=0.004)
  qda.fit(Xbs, Tbs)
  prob_matrix = qda.predict_proba(Xval) 
  averaged_prob_matrix = averaged_prob_matrix + prob_matrix

  predicted_class = np.argmax(averaged_prob_matrix/(i+1), axis=1)
  bagged_val_acc = 1-(np.count_nonzero(predicted_class - Tval)/Xval.shape[0])

  all_val_acc.append(bagged_val_acc)

plt.plot(all_i, all_val_acc)
plt.title("Validation accuracy")
plt.xlabel("Number of bootstrap samples")
plt.ylabel("Accuracy")
plt.show()

plt.semilogx(all_i, all_val_acc)
plt.title("Validation accuracy (log scale)")
plt.xlabel("Number of bootstrap samples")
plt.ylabel("Accuracy")
plt.show()

def train3d(K,R,X,T):
  pca = PCA(n_components=K, svd_solver='full')
  pca.fit(X)
  Xreduced = pca.transform(X)
  qda = QuadraticDiscriminantAnalysis(reg_param=R)
  qda.fit(Xreduced, T)
  return pca, qda

def proba3d(pca,qda,X):
  Xreduced = pca.transform(X)
  prob_matrix = qda.predict_proba(Xreduced)
  return prob_matrix

def myBag(K,R):
  pca, qda = train3d(K, R, Xsmall, Tsmall)
  Xval_reduced = pca.transform(Xval)
  base_val_acc = qda.score(Xval_reduced, Tval)
  averaged_prob_matrix = np.zeros((Xval.shape[0], 10))
  
  for i in range(0, 200): 
    Xbs, Tbs = myBootstrap(Xsmall, Tsmall)
    pca, qda = train3d(K, R, Xbs, Tbs)
    prob_matrix = proba3d(pca, qda, Xval)
    averaged_prob_matrix = averaged_prob_matrix + prob_matrix

  averaged_prob_matrix = averaged_prob_matrix/200
  predicted_class = np.argmax(averaged_prob_matrix, axis=1)
  bagged_val_acc = 1-(np.count_nonzero(predicted_class - Tval)/Xval.shape[0])

  return base_val_acc, bagged_val_acc

print('-------------')

base_val_acc, bagged_val_acc = myBag(100, 0.01)
print("\nBase Validation Acc: %.4f Bagged Validation Acc: %.4f" % (base_val_acc, bagged_val_acc))

all_base_acc = []
all_bagged_acc = []
for i in range(0, 50):
  K = random.randint(1, 10)
  R = random.uniform(0.2, 1.0)
  base_val_acc, bagged_val_acc = myBag(K, R)
  all_bagged_acc.append(bagged_val_acc)
  all_base_acc.append(base_val_acc)

plt.scatter(all_base_acc, all_bagged_acc, color='blue')
plt.plot([0,1], [0,1], color='red')
plt.title("Bagged v.s. base validation accuracy")
plt.xlabel("Base validation accuracy")
plt.ylabel("Bagged validation accuracy")
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()

print('-------------')
all_base_acc = []
all_bagged_acc = []
for i in range(0, 50):
  K = random.randint(50, 200)
  R = random.uniform(0, 0.05)
  base_val_acc, bagged_val_acc = myBag(K, R)
  all_bagged_acc.append(bagged_val_acc)
  all_base_acc.append(base_val_acc)

plt.scatter(all_base_acc, all_bagged_acc, color='blue')
height = max(all_bagged_acc)
plt.plot([min(all_base_acc),max(all_base_acc)], [height,height], color='red')
plt.title("Bagged v.s. base validation accuracy")
plt.xlabel("Base validation accuracy")
plt.ylabel("Bagged validation accuracy")
plt.ylim(0, 1)
plt.show()

print("\nMaximum Bagged Validation Acc: %.4f" % (height))