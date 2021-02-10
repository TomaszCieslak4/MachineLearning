import matplotlib.pyplot as plt
import sklearn.linear_model as lin
from scipy.stats import multivariate_normal
import pickle
import time
import math
import numpy as np
import numpy.random as rnd

with open("data.pickle", "rb") as file:
    dataTrain, dataTest = pickle.load(file)
    Xtrain, Ttrain = dataTrain
    Xtest, Ttest = dataTest

print("----------")

from sklearn.neural_network import MLPClassifier

def setup_NN(seed, hidden_units, training_itr): # Setup a single layer neural network given the parameters
    """
    Usage:
    setup_NN(0, 1, 1000)
    """
    np.random.seed(seed)

    with open("data.pickle", "rb") as file:
        dataTrain, dataTest = pickle.load(file)
        Xtrain, Ttrain = dataTrain
        Xtest, Ttest = dataTest
    
    clf = MLPClassifier(hidden_layer_sizes=(hidden_units, ), tol=1e-05, learning_rate_init=0.01,
                        solver='sgd', activation='logistic', max_iter=training_itr)
    clf.fit(Xtrain, Ttrain)
    accuracy = clf.score(Xtest, Ttest) # Test Accuracy


def accuracyNN(clf, X, T):
    """
    Returns the accuracy of classifier clf on data X,T, where clf is a neural network with one hidden layer using logistic activation function 
    NOTE: Do not use any methods of MLPClassifier or sklearn
    """    
    first_hidden_layer = np.matmul(X, clf.coefs_[0]) + clf.intercepts_[0]
    first_hidden_layer = 1/(1+np.exp(-1*first_hidden_layer))
    final_layer = np.matmul(first_hidden_layer, clf.coefs_[1] ) + clf.intercepts_[1] 
    final_layer = 1/(1+np.exp(-1*final_layer))
    predicted_class = np.argmax(final_layer, axis=1)
    return 1-(np.count_nonzero(predicted_class - T)/X.shape[0])

np.random.seed(0)
clf4 = MLPClassifier(hidden_layer_sizes=(9, ), tol=1e-05, learning_rate_init=0.01,
                    solver='sgd', activation='logistic', max_iter=1000)
clf4.fit(Xtrain, Ttrain)
accuracy1 = clf4.score(Xtest, Ttest) # Test Accuracy
accuracy2 = accuracyNN(clf4, Xtest, Ttest)
print("\naccuracy1: ", accuracy1, "accuracy2: ", accuracy2, "difference: ", accuracy1-accuracy2)

def ceNN(clf,X,T):
    """
    Computes CE1 and CE2 for classifier clf on data X,T, where clf is a neural net with one hidden layer.
    NOTE: CE1 can use methods from clf
          CE2 cannot use any methods from clf
    """  
    # Compute the one hot vector for all classes and store it as a matrix
    one_hot_matrix = np.zeros((T.size, T.max()+1))
    one_hot_matrix[np.arange(T.size),T] = 1

    # Compute CE1
    log_prob1 = -clf.predict_log_proba(X) 
    CE1_matrix = np.multiply(one_hot_matrix , log_prob1) + np.multiply((1-one_hot_matrix) , -log_prob1)
    CE1_matrix = np.multiply(CE1_matrix, one_hot_matrix)
    CE1 = np.sum(CE1_matrix)/X.shape[0]

    # Create the final layer of the neural net
    first_hidden_layer = np.matmul(X, clf.coefs_[0]) + clf.intercepts_[0]
    first_hidden_layer = 1/(1+np.exp(-1*first_hidden_layer))
    final_layer = np.matmul(first_hidden_layer, clf.coefs_[1] ) + clf.intercepts_[1] 
    final_layer = np.exp(final_layer)
    final_layer = np.divide(final_layer, np.sum(final_layer, axis=1).reshape(X.shape[0],1))

    # Compute CE2
    log_prob2 = -np.log(final_layer)
    CE2_matrix = np.multiply(one_hot_matrix , log_prob2) + np.multiply((1-one_hot_matrix) , -log_prob2)
    CE2_matrix = np.multiply(CE2_matrix, one_hot_matrix)
    CE2 = np.sum(CE2_matrix)/X.shape[0]
    return (CE1,CE2)

np.random.seed(0)
clf5 = MLPClassifier(hidden_layer_sizes=(9, ), tol=1e-05, learning_rate_init=0.01,
                    solver='sgd', activation='logistic', max_iter=1000)
clf5.fit(Xtrain, Ttrain)
(CE1, CE2) = ceNN(clf5, Xtest, Ttest)
print("\nCE1: ", CE1, "CE2: ", CE2, "difference: ", CE1-CE2)

print("----------")

# get the mnist data from a file
with open('mnistTVT.pickle','rb') as f:
    Xtrain,Ttrain,Xval,Tval,Xtest,Ttest = pickle.load(f)

# Construct reduced data consisting of digits d1 and d2.
# all data sets are global variables.
def reduce(d1,d2):
    global Xtrain2, Ttrain2, Xtest2, Ttest2
    # reduced training data
    idx = (Ttrain==d1) | (Ttrain==d2) # index to digits d1 and d2
    Xtrain2 = Xtrain[idx]
    Ttrain2 = Ttrain[idx]
    # reduced test data
    idx = (Ttest==d1) | (Ttest==d2)
    Xtest2 = Xtest[idx]
    Ttest2 = Ttest[idx]

reduce(5,6)

def relabel_56(x):
    if x == 5:
        return 1
    else:
        return 0

def relabel_45(x):
    if x == 4:
        return 1
    else:
        return 0


Ttrain2 = np.vectorize(relabel_56)(Ttrain2)
Ttest2 = np.vectorize(relabel_56)(Ttest2)

def evaluateNN(clf,X,T):
    accuracy1, accuracy2, CE1, CE2 = 0, 0, 0, 0

    def threshold(x):
        if x > 0.5:
            return 1
        else:
            return 0

    T = T.reshape(X.shape[0],1) # Reshape T

    # Compute Accuracy2
    accuracy1 = clf.score(X, T) # Test Accuracy

    # Compute Accuracy2
    h = np.matmul(X, clf.coefs_[0]) + clf.intercepts_[0]
    h = np.tanh(h)

    g = np.matmul(h, clf.coefs_[1]) + clf.intercepts_[1]
    g = np.tanh(g)

    o = np.matmul(g, clf.coefs_[2]) + clf.intercepts_[2]
    o = (1/(1+np.exp(-1*o)))

    predicted_o = np.vectorize(threshold)(o)
    accuracy2 = 1-(np.count_nonzero(predicted_o - T)/X.shape[0])

    # Compute CE1
    log_prob1 = clf.predict_log_proba(X)
    CE1 = np.mean(-np.multiply(T , log_prob1[:,1:2]) - np.multiply((1-T) , log_prob1[:,0:1]))

    # Compute CE2
    h2 = np.matmul(X, clf.coefs_[0]) + clf.intercepts_[0]
    h2 = np.tanh(h2)

    g2 = np.matmul(h2, clf.coefs_[1]) + clf.intercepts_[1]
    g2 = np.tanh(g2)

    o2 = np.matmul(g2, clf.coefs_[2]) + clf.intercepts_[2]
    o2 = (1/(1+np.exp(-1*o2)))

    CE2 = np.mean(-np.multiply(T , np.log(o2)) - np.multiply((1-T) , np.log(1-o2)))

    return(accuracy1, accuracy2, CE1, CE2)

np.random.seed(0)
clf6 = MLPClassifier(hidden_layer_sizes=(100,100), solver='sgd', activation="tanh",
                            learning_rate_init=0.01, tol=10 ** (-6), max_iter=100, batch_size=100)
clf6.fit(Xtrain2, Ttrain2)
(accuracy1, accuracy2, CE1, CE2) = evaluateNN(clf6, Xtest2, Ttest2)
print('accuracy1 : ', accuracy1, 'accuracy2 : ', accuracy2, '\nCE1 : ', CE1, 'CE2 : ', CE2)

acc2_list = []
CE2_list = []
batch_list = []
for i in range(0, 14):
    np.random.seed(0)
    clf7 = MLPClassifier(hidden_layer_sizes=(100,100), solver='sgd', activation="tanh",
                            learning_rate_init= 0.001, tol=10 ** (-6), max_iter=1, batch_size=2**i)
    clf7.fit(Xtrain2, Ttrain2)
    (accuracy1, accuracy2, CE1, CE2) = evaluateNN(clf7, Xtest2, Ttest2)
    acc2_list.append(accuracy2)
    CE2_list.append(CE2)
    batch_list.append(2**i)

plt.plot(batch_list, acc2_list)
plt.semilogx()
plt.title('Accuracy v.s. batch size')
plt.xlabel('batch size')
plt.ylabel('accuracy')
plt.show()

plt.plot(batch_list, CE2_list)
plt.semilogx()
plt.title('Cross entropy v.s. batch size')
plt.xlabel('batch size')
plt.ylabel('cross entropy')
plt.show()
