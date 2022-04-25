import numpy as np
import logging
from math import sqrt
import time

logger = logging.get_logger(__name__)

class naiveBayes:
    """A basic class to define all common Naive Bayes-related methods.

    References:
        
    """

    def __init__(self):

        logger.info('Creating class: naiveBayes.')

    
    # Fit the training data
    def fit(self, X, y):
        """Fits data in the classifier.

        Args:
            X (np.array): Array of training features.
            y (np.array): Array of training labels.
        
        """

        start = time.time()

        dataset = np.column_stack((X, y))

        model = summarize_by_class(dataset)

        

        end = time.time()



