import numpy as np
import logging
from math import sqrt, pi, exp
import time
from utils.spliting import summarize_by_class

# Functio of gaussian probability
def gaussian_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= gaussian_probability(row[i], mean, stdev)
    return probabilities

class GaussianNB:
    """A basic class to define Gaussian Naive Bayes-related methods.

    References:
        
    """

    def __init__(self, distribution_type='Gaussian'):
        """Initialization method.

        """ 

        self.summaries = None
        self.train_time = None
        self.test_time = None
        self.score_time = None
    
    def fit(self, X, y):
        """Fits data in the classifier.

        Args:
            X (np.array): Array of training features.
            y (np.array): Array of training labels.
        
        """

        start = time.time()

        dataset = np.column_stack((X, y))

        self.summaries = summarize_by_class(dataset)

        end = time.time()

        self.train_time = end - start


    # Predict the class for a given row
    def predict(self, X):
        """Predict the data with the fitted dataset.

        Args:
            X (np.array): Array of test features.
        
        Return:
            y (np.array): Array of predict class

        """

        start = time.time()

        y = np.zeros(len(X))

        if self.summaries == None:
            print("The dataset must be fitted")
            return None
        else:
            for j in range(len(X)):
                probabilities = calculate_class_probabilities(self.summaries, X[j])
                best_label, best_prob = None, -1
                for class_value, probability in probabilities.items():
                    if best_label is None or probability > best_prob:
                        best_prob = probability
                        best_label = class_value
                y[j] = best_label
            
        end = time.time()

        self.test_time = end - start

        return y

    def scores(self, X):
        """Return the scores from the predicted data with the fitted dataset.

        Args:
            X (np.array): Array of test features.
        
        Return:
            scores (np.array): Array of predicted data's scores 

        """

        start = time.time()

        scores = np.zeros(len(X))

        if self.summaries == None:
            print("The dataset must be fitted")
            return None
        else:
            for j in range(len(X)):
                probabilities = calculate_class_probabilities(self.summaries, X[j])
                best_label, best_prob = None, -1
                for class_value, probability in probabilities.items():
                    if best_label is None or probability > best_prob:
                        best_prob = probability
                        best_label = class_value
                
                # Gets the best probability and divide by the sum of all giving a normalized score
                scores[j] = best_prob/(sum(probabilities.values()))
            
        end = time.time()

        self.score_time = end - start

        return scores




