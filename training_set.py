# -*- coding: utf-8 -*-
"""
Created on Fri Apr 2 2021

@author: Vinicius Lima
"""

import numpy as np
import matplotlib.image as mpimg

class training_test_sets():
    
    def __init__(self, num_samples, im_set_path):
        
        self.num_samples = num_samples
        self.num_test_samples = 10 - num_samples
        self.im_set_path = im_set_path

        self.X_train = np.zeros((10304,40*num_samples))
        self.y_train = np.zeros(40*num_samples)
        self.X_test = np.zeros((10304,40*self.num_test_samples))
        self.y_test = np.zeros(40*self.num_test_samples)

        
    def solve(self):

        for ii in range(40):  # iterating over subfolders
            for jj in range(self.num_samples):
                im_path = self.im_set_path + '/s' + str(ii+1) + '/' + str(jj+1) + '.pgm'
                cur_img = mpimg.imread(im_path)
                self.X_train[:, ii*self.num_samples + jj] = np.ravel(cur_img)
                self.y_train[ii*self.num_samples + jj] = int(ii + 1)
            for jj in range(self.num_test_samples):
                im_path = self.im_set_path + '/s' + str(ii+1) + '/' + str(jj+10) + '.pgm'
                cur_img = mpimg.imread(im_path)
                self.X_test[:, ii*self.num_test_samples + jj] = np.ravel(cur_img)
                self.y_test[ii*self.num_test_samples + jj] = int(ii + 1)
        
        return self.X_train, self.y_train, self.X_test, self.y_test