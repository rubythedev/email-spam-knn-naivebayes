'''knn.py
K-Nearest Neighbors algorithm for classification
Ruby Nunez
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from palettable import cartocolors


class KNN:
    '''K-Nearest Neighbors supervised learning algorithm'''

    def __init__(self, num_classes):
        '''KNN constructor
        '''

        self.num_classes = num_classes
        # exemplars: ndarray. shape=(num_train_samps, num_features).
        #   Memorized training examples
        self.exemplars = None
        # classes: ndarray. shape=(num_train_samps,).
        #   Classes of memorized training examples
        self.classes = None


    def train(self, data, y):
        '''Train the KNN classifier on the data `data`, where training samples have corresponding
        class labels in `y`.

        Parameters:
        -----------
        data: ndarray. shape=(num_train_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_train_samps,). Corresponding class of each data sample.
        '''

        self.exemplars = data
        self.classes = y


    def predict(self, data, k):
        '''Use the trained KNN classifier to predict the class label of each test sample in `data`.
        Determine class by voting: find the closest `k` training exemplars (training samples) and
        the class is the majority vote of the classes of these training exemplars.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions.

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_test_samps,). Predicted class of each test data
        sample.
        '''

        num_test_samps = data.shape[0]
        predicted_classes = np.zeros(num_test_samps, dtype=int)

        for i in range(num_test_samps):
            distances = np.sqrt(np.sum((self.exemplars - data[i])**2, axis=1))
            closest_indices = np.argsort(distances)[:k]
            class_counts = np.zeros(self.num_classes, dtype=int)
            for j in range(k):
                class_counts[int(self.classes[closest_indices[j]])] += 1
            max_count = -1
            max_class = -1
            for j in range(self.num_classes):
                if class_counts[j] > max_count:
                    max_count = class_counts[j]
                    max_class = j

            predicted_classes[i] = max_class

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

        num_correct = np.sum(y == y_pred)
        total_samples = y.shape[0]
        accuracy = num_correct / total_samples

        return accuracy
    

    def plot_predictions(self, k, n_sample_pts):
        '''Paints the data space in colors corresponding to which class the classifier would
         hypothetically assign to data samples appearing in each region.

        Parameters:
        -----------
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions.
        n_sample_pts: int.
            How many points to divide up the input data space into along the x and y axes to plug
            into KNN at which we are determining the predicted class.
        '''

        color_palette = cartocolors.qualitative.Safe_4.mpl_colors
        cmap = ListedColormap(color_palette)
        samp_vec = np.linspace(-40, 40, n_sample_pts)
        x, y = np.meshgrid(samp_vec, samp_vec)
        data = np.column_stack((x.flatten(), y.flatten()))
        y_pred = self.predict(data, k)
        y_pred_grid = y_pred.reshape((n_sample_pts, n_sample_pts))

        plt.pcolormesh(x, y, y_pred_grid, cmap=cmap)
        plt.colorbar()

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