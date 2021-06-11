"""
Created on Fri Apr 2 2021

@author: Vinicius Lima
"""

import numpy as np
import scipy.sparse.linalg

class training_set_PCA():
    
    def __init__(self, X_train, k):

        self.X_train = X_train
        self.k = k

    def solve(self):

        N, n_faces = self.X_train.shape
        mu = self.X_train.mean(axis=1) 
        X = np.zeros((N, n_faces))
        for ii in range(n_faces):
            X[:, ii] = self.X_train[:, ii] - mu

        sigma = np.matmul(X, X.transpose()) 
        sigma /= n_faces
        _, sigma_eigs = scipy.sparse.linalg.eigs(sigma, k=self.k)

        # Projecting faces
        P_H = np.conj(np.transpose(sigma_eigs))
        X_pca = np.matmul(P_H, X)

        return mu, sigma_eigs, X_pca

class test_set_PCA():
    def __init__(self, mu, P, X_test):

        self.X_test = X_test
        self.mu = mu
        self.P = P

    def solve(self):

        N, n_faces = self.X_test.shape
        X = np.zeros((N, n_faces))

        for ii in range(n_faces):
            X[:, ii] = self.X_test[:, ii] - self.mu
    
        P_H = np.conj(np.transpose(self.P))
        X_pca = np.matmul(P_H, X)

        return X_pca