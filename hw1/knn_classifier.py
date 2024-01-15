import numpy as np
from collections import defaultdict


class KNNClassifier():
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.points = np.array(X_train)
        self.labels = np.array(y_train)

    def predict(self, new_points, order=2):
        res = []
        fmap = defaultdict(int)
        for new_point in new_points:
            distances = np.linalg.norm(self.points - new_point, ord=order, axis=1)
            k_min_indices = np.argpartition(distances, self.k)[:self.k]

            for idx in k_min_indices:
                fmap[tuple(self.labels[idx])] += 1
            res.append(max(fmap.items(), key=lambda x: x[1])[0])
            fmap.clear()
        return np.array(res)

    def evaluate(self, test_points, test_labels, order=2, verbose=True):
        train_predictions = self.predict(self.points, order=order)
        test_predictions = self.predict(test_points, order=order)
        train_accuracy = np.mean(train_predictions == self.labels)
        test_accuracy = np.mean(test_predictions == test_labels)
        train_error = 1 - train_accuracy
        test_error = 1 - test_accuracy
        train_positive_rate = np.mean(train_predictions == np.array([0, 1]))
        test_positive_rate = np.mean(test_predictions == np.array([0, 1]))
        if verbose:
            print("Train Accuracy: {:.3f}".format(train_accuracy))
            print("Train Error: {:.3f}".format(train_error))
            print("Train Positive Rate: {:.3f}".format(train_positive_rate))
            print("Test Accuracy: {:.3f}".format(test_accuracy))
            print("Test Error: {:.3f}".format(test_error))
            print("Test Positive Rate: {:.3f}".format(test_positive_rate))
        return train_accuracy, train_error, train_positive_rate, test_accuracy, test_error, test_positive_rate
