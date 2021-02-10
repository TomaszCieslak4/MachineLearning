import matplotlib.pyplot as plt
import pickle
import time
import math
import numpy as np
import numpy.random as rnd


print('-------------')
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

with open("data.pickle", "rb") as file:
    dataTrain, dataTest = pickle.load(file)
    Xtrain, Ttrain = dataTrain
    Xtest, Ttest = dataTest

def plot_clusters(X,R,Mu):
  sums_columnwise = np.sum(R, axis=0) # first add up the total responsibility in each column of R
  R = R[:,np.argsort(sums_columnwise)] # sort the columns of R in ascending order

  # This takes the NxK weights and turns then into percentages rowwise
  sum_rowwise = np.sum(R, axis=1)
  R = R/sum_rowwise[:,None]

  # X is still alligned with R but columns might be shifted around
  plt.scatter(X[:, 0], X[:, 1], color=R, s=5) # Plot all X with its R
  plt.scatter(Mu[:, 0], Mu[:, 1], color='black') # Plot all center of clusters

print('-------------')

km = KMeans(n_clusters=3)
km.fit(Xtrain)

# These are class predictions
labels = km.labels_
responsibility = np.zeros((Xtrain.shape[0], 3))
responsibility[np.arange(Xtrain.shape[0]), labels] = 1

plot_clusters(Xtrain, responsibility ,km.cluster_centers_)
plt.title("K means")
plt.show()

km_train_score = km.score(Xtrain)
km_test_score = km.score(Xtest)
print("\nTrain Score: %.4f Test Score: %.4f" % (km_train_score, km_test_score))

print('-------------')

gm = GaussianMixture(covariance_type='spherical', n_components=3)
gm.fit(Xtrain)

# These are class predictions
responsibility = gm.predict_proba(Xtrain)

plot_clusters(Xtrain, responsibility ,gm.means_)
plt.title("Gaussian mixture model (spherical)")
plt.show()

gm1_train_score = gm.score(Xtrain)
gm1_test_score = gm.score(Xtest)
print("\nTrain Score: %.4f Test Score: %.4f" % (gm1_train_score, gm1_test_score))

print('-------------')

gm = GaussianMixture(covariance_type='full', n_components=3)
gm.fit(Xtrain)

# These are class predictions
responsibility = gm.predict_proba(Xtrain)

plot_clusters(Xtrain, responsibility ,gm.means_)
plt.title("Gaussian mixture model (full)")
plt.show()

gm2_train_score = gm.score(Xtrain)
gm2_test_score = gm.score(Xtest)
print("\nTrain Score: %.4f Test Score: %.4f" % (gm2_train_score, gm2_test_score))
print("\n(Mine vs. them) Difference in gm test scores = %.4f" % (gm2_test_score - gm1_test_score))

def scoreKmeans(X,Mu):
  all_dist = np.ones((X.shape[0], Mu.shape[0]))
  for n in range(Mu.shape[0]):
      all_dist[:,n] = np.sum(np.square(X-Mu[n]),axis=1)
  return np.sum(np.min(all_dist, axis=1))

def myKmeans(X,K,I):
  '''
  Returns K clusters in data matrix X by performing I iterations of the (hard) K-means algorithm
  '''
  random_indices = np.random.choice(X.shape[0], size=K, replace=False) # Choose K random points uniformly from X without replacement
  Mu = X[random_indices] # Set Mu to the K random points in X
  scores = []
  all_dist = np.ones((X.shape[0], K))

  for i in range(I): #  (hard) K-means algorithm
    R = np.zeros((X.shape[0], K))
    scores.append(scoreKmeans(X,Mu))
    all_dist = np.sqrt(np.sum(np.square(X[:,np.newaxis]-Mu),axis=2))
    R[np.arange(R.shape[0]), np.argmin(all_dist, axis=1)] = 1
    classes = np.argmin(all_dist, axis=1)
    split_x=np.repeat(R.T[:,:,np.newaxis],X.shape[1],axis=2)
    Mu=np.sum(X*split_x,axis=1)/np.sum(R,axis=0)[:, np.newaxis]

  return Mu, R, scores

print('-------------')

train_Mu, train_R, train_scores = myKmeans(Xtrain, 3, 100)
test_Mu, test_R, test_scores = myKmeans(Xtest, 3, 100)

train_s = scoreKmeans(Xtrain,train_Mu)
test_s = scoreKmeans(Xtest,test_Mu)

print("\nTrain Score: %.4f Test Score: %.4f" % (train_s, test_s))

print("\n(Mine vs. them) Difference in kmeans test scores = %.4f" % (test_s - km_test_score))

plt.plot(np.arange(20), train_scores[:20])
plt.title("score v.s. iteration (K means)")
plt.xlabel("Iteration")
plt.ylabel("Score")
plt.show()

plot_clusters(Xtrain, train_R, train_Mu)
plt.title("Data clustered by K means")
plt.show()
