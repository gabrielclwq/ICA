from math import sqrt

def mean(numbers):
    """Calculate the mean value of a list.

        Args:
            numbers (list): List of numbers.
        Returns:
            mean (float): Mean value of the list of numbers.

    """

    return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
    """Calculate the standard deviation of a list.

        Args:
            numbers (list): List of numbers.
        Returns:
            deviation (float): Standard deviation of the list of numbers.

    """

    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    deviation = sqrt(variance)
    return deviation

# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
    """Calculate the mean value of a list.

        Args:
            dataset (np.array): Array of values [features labels].
                dataset = np.column_stack((X, y))
                X (np.array): Array of training features
                y (np.array): Array of training labels 
        Returns:
            summaries (list): List of mean, standard deviation and length for each feature column in dataset.

    """

    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del(summaries[-1])
    return summaries