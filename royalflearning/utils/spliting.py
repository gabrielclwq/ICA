
from statistics.basics import summarize_dataset

def separate_by_class(dataset):
    """Separate the dataset by class.

        Args:
            X (np.array): Array of training features.
        Returns:
            separated (dictionary): Dict of separated dataset (keys = class, values = features).
    """

    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated

def summarize_by_class(dataset):
    """Separate the dataset by class and calculate the mean, standard deviation and length for each row.

        Args:
            dataset (np.array): Array of values [features labels].
                dataset = np.column_stack((X, y))
                X (np.array): Array of training features
                y (np.array): Array of training labels 
        Returns:
            summaries (dictionary): Dict of separated dataset (keys = class, values = mean, standard deviation and length).
    """

    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries