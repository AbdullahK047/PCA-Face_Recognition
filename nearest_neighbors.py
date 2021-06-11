"""
Created on Fri Apr 2 2021

@author: Vinicius Lima
"""

import numpy as np
import scipy.spatial.distance

class nearest_neighbors_classifier():
    
    def __init__(self, X_train, X_test):

        self.X_train = X_train
        self.X_test = X_test

    def solve(self):
        distances = scipy.spatial.distance.cdist(self.X_train.transpose(), self.X_test.transpose())
        predicted_labels = np.argmin(distances, axis=0)

        return predicted_labels