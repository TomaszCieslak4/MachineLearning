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

print('-------------')
print('*** Using only 5,000 MNIST training points (for improved performance)***')
Xtrain = Xtrain[0:5000]

from sklearn.decomposition import PCA

def transform(n):
    pca = PCA(n_components=n)
    pca.fit(Xtrain)
    reduced_data = pca.transform(Xtest)
    projected_data = pca.inverse_transform(reduced_data) 
    fig, grid = plt.subplots(5, 5)
    for i in range(25):
        image = projected_data[i].reshape((28,28))
        grid[i//5, i%5].imshow(image, interpolation='nearest', cmap='Greys')
        grid[i//5, i%5].axis('off')
    plt.suptitle('MNIST test data projected onto {} dimensions'.format(n))

transform(30)
plt.show()
transform(3)
plt.show()
transform(300)
plt.show()

def myPCA(X,K):

  means = np.mean(X , axis = 0)
  X_meaned = X - means
  cov_mat = np.dot(X_meaned.T, X_meaned) / (X.shape[0])
  
  eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
  eigenvector_subset = eigen_vectors[:,-K:]

  X_reduced = np.matmul(eigenvector_subset.T, X_meaned.T)
  Xp = np.matmul(eigenvector_subset, X_reduced).T + means
  return Xp

print('-------------')
myXtrainP =myPCA(Xtrain, 100)
fig, grid = plt.subplots(5, 5)
for i in range(25):
  image = myXtrainP[i].reshape((28,28))
  grid[i//5, i%5].imshow(image, interpolation='nearest', cmap='Greys')
  grid[i//5, i%5].axis('off')
plt.suptitle('MNIST data projected onto 100 dimensions (my implementation)')
plt.show()

pca = PCA(n_components=100, svd_solver='full')
pca.fit(Xtrain)
XtrainR = pca.transform(Xtrain)
XtrainP = pca.inverse_transform(XtrainR) 

fig, grid = plt.subplots(5, 5)
for i in range(25):
  image = XtrainP[i].reshape((28,28))
  grid[i//5, i%5].imshow(image, interpolation='nearest', cmap='Greys')
  grid[i//5, i%5].axis('off')
plt.suptitle('MNIST data projected onto 100 dimensions (sklearn)')
plt.show()

rms = np.mean(np.square(XtrainP - myXtrainP))**(1/2)
print('\nRMS = ',rms)