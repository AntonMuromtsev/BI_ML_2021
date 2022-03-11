import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        distances = np.zeros(X.shape[0] * self.train_X.shape[0]).reshape(X.shape[0], self.train_X.shape[0])
        for i in np.arange(X.shape[0]):
            for j in np.arange(self.train_X.shape[0]):
                distances[i,j] = np.sum(np.absolute(X[i] - self.train_X[j]))

        return distances






    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        """
        YOUR CODE IS HERE
        """
        distances = np.zeros(X.shape[0] * self.train_X.shape[0]).reshape(X.shape[0], self.train_X.shape[0])
        for i in np.arange(X.shape[0]):
            future_row = np.absolute(X[i] - self.train_X)
            distances[i] = future_row.sum(axis = 1)
        return distances


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        distances = np.sum(np.abs(X[:, np.newaxis] - self.train_X), axis=2)
        return distances




    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test)

        """
        Solution for k = 1
        pred = np.zeros(distances.shape[0])
        for i in np.arange(distances.shape[0]):
            index = np.where(distances[i] == np.min(distances[i]))[0][0]
            pred[i] = self.train_y[index]
        return pred
        """
        pred = np.zeros(distances.shape[0]).astype(str)
        for i in np.arange(distances.shape[0]):
            indices = np.argpartition(distances[i], self.k)
            values = self.train_y[indices[:self.k]]
            plus = np.sum(values == "1")
            minus = np.sum(values == "0")
            if plus > minus:
                pred[i] = "1"
            else:
                pred[i] = "0"
        return pred


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        n_train = distances.shape[0]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test, np.int)

        pred = np.zeros(distances.shape[0]).astype(str)
        for i in np.arange(distances.shape[0]):
            indices = np.argpartition(distances[i], self.k)
            values = self.train_y[indices[:self.k]]
            scores = np.zeros(10)
            for j in range(10):
                scores[j] = np.sum(values == f"{j}")
            pred[i] = np.argmax(scores)
        return pred

