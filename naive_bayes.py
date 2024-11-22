'''naive_bayes.py
Naive Bayes classifier with Multinomial likelihood for discrete features
Ruby Nunez
'''

import numpy as np


class NaiveBayes:
    '''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any
     number of classes)'''
    def __init__(self, num_classes):
        '''Naive Bayes constructor'''

        self.num_classes = num_classes

        # class_priors: ndarray. shape=(num_classes,).
        #   Probability that a training example belongs to each of the classes
        #   For spam filter: prob training example is spam or ham
        self.class_priors = None

        # class_likelihoods: ndarray. shape=(num_classes, num_features).
        #   Probability that each word appears within class c
        self.class_likelihoods = None


    def get_priors(self):
        '''Returns the class priors (or log of class priors if storing that)'''

        return self.class_priors


    def get_likelihoods(self):
        '''Returns the class likelihoods (or log of class likelihoods if storing that)'''

        return self.class_likelihoods


    def train(self, data, y):
        '''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
        class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
        class likelihoods (the probability of a word appearing in each class — spam or ham)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.
        '''

        num_samples, num_features = data.shape
        
        unique_classes, class_counts = np.unique(y, return_counts=True)
        self.class_priors = class_counts / num_samples
        
        self.class_likelihoods = np.zeros((self.num_classes, num_features))
        for i in range(self.num_classes):
            class_data = data[y == unique_classes[i]]
            feature_likelihoods = (class_data.sum(axis=0) + 1) / (class_data.shape[0] + 2)
            self.class_likelihoods[i] = feature_likelihoods
        
        self.class_likelihoods /= np.sum(self.class_likelihoods, axis=1, keepdims=True)


    def predict(self, data):
        '''Combine the class likelihoods and priors to compute the posterior distribution. The
        predicted class for a test sample from `data` is the class that yields the highest posterior
        probability.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each test data sample.
        '''
        
        num_test_samples = data.shape[0]

        log_posterior = np.zeros((num_test_samples, self.num_classes))
        for i in range(self.num_classes):
            log_posterior[:, i] = np.log(self.class_priors[i]) + (data * np.log(self.class_likelihoods[i])).sum(axis=1)

        predicted_classes = np.argmax(log_posterior, axis=1)

        return predicted_classes


    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.
        '''

        accuracy = np.mean(y == y_pred) * 100

        return accuracy


    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        Parameters:
        -----------
        y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        '''

        num_classes = self.num_classes
        confusion_matrix = np.zeros((num_classes, num_classes))
        
        for i in range(num_classes):
            for j in range(num_classes):
                confusion_matrix[i, j] = np.sum((y == i) & (y_pred == j))
        
        return confusion_matrix