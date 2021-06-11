# -*- coding: utf-8 -*-
"""
Created on Fri Apr 2 2021

@author: Vinicius Lima
"""

###############################################################################
############################# I M P O R T I N G ###############################
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from training_set import training_test_sets
from PCAs import training_set_PCA, test_set_PCA
from nearest_neighbors import nearest_neighbors_classifier

###############################################################################
################################## M A I N ####################################
###############################################################################

if __name__ == '__main__': 
    # Creating training and test sets
    n_samples = 9
    im_dataset_path = 'att_faces'
    training_set_helper = training_test_sets(n_samples, im_dataset_path)

    training_set, training_labels, test_set, test_labels = training_set_helper.solve()

    k_list = [1, 5, 10, 20]
    accuracies = np.zeros(len(k_list))
    ii = 0
    n_examples = 3

    for k in k_list:
        # PCA transform of training points
        train_PCA = training_set_PCA(training_set, k)
        training_mean, PCA_mtx, training_PCAs = train_PCA.solve()

        # PCA transform of test samples
        test_PCA = test_set_PCA(training_mean, PCA_mtx, test_set)
        test_PCAs = test_PCA.solve()

        # Nearest neighbors
        NN_classifier = nearest_neighbors_classifier(training_PCAs, test_PCAs)
        predicted_labels = NN_classifier.solve()

        # Classification accuracy
        accuracy = np.mean(training_labels[predicted_labels] == test_labels)
        accuracies[ii] = accuracy
        ii += 1

        for jj in range(n_examples):
            # Displaying images
            im_number = np.random.randint(1, 40)  # random image from the test set
            im_path = im_dataset_path + '/s' + str(im_number) + '/' + str(10) + '.pgm'
            og_img = mpimg.imread(im_path)
            predicted_person = int(training_labels[predicted_labels[im_number-1]])
            s_number = predicted_labels[im_number - 1] - (predicted_person-1)*n_samples
            im_path = im_dataset_path + '/s' + str(predicted_person) + '/' + str(s_number+1) + '.pgm'
            nearest_img = mpimg.imread(im_path)
            fig = plt.figure()
            plt.suptitle('Original and Nearest Images: k = ' + str(k), fontsize=10)
            ax1 = fig.add_subplot(1,2,1)
            ax1.imshow(og_img, cmap='gray')
            ax1.axes.xaxis.set_visible(False)
            ax1.axes.yaxis.set_visible(False)
            plt.title('Original', fontsize=9)
            ax2 = fig.add_subplot(1,2,2)
            ax2.imshow(nearest_img, cmap='gray')
            ax2.axes.xaxis.set_visible(False)
            ax2.axes.yaxis.set_visible(False)
            plt.title('Nearest', fontsize=9)
            # Saving plot
            fig_path_name = 'k_' + str(k) + '_ex_' + str(jj) + '.png'
            plt.savefig(fig_path_name, dpi=300)
            plt.show()

    # Plotting classification accuracy
    acc_plt = plt.figure()
    ax = acc_plt.add_subplot(111)
    ax.plot(np.asarray(k_list), accuracies, 'tab:blue', marker='o', linestyle='dashed', linewidth=1.2, markersize=6)
    ax.grid(linestyle='-.')
    plt.title('Nearest Neighbors: Classification Accuracy', fontsize=10)
    plt.xlabel('Number of Principal Components', fontsize=9)
    plt.ylabel('Accuracy', fontsize=9)
    plt.xticks(k_list, fontsize=8)
    plt.yticks(np.arange(0, 1, step=0.1), fontsize=8)
    plt.xlim([0, k_list[-1]+1])
    plt.ylim([0.0, 1.])
    ratio = 1/(16/9)
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    # Saving plot
    fig_path_name = 'nearest_neighbors_accuracy.png'
    plt.savefig(fig_path_name, dpi=300)
    plt.show()





