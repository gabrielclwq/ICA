import pandas as pd
import numpy as np

data = pd.read_csv("Iris.csv")
print(data.head())

y = data["Species"]
X = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]

# Get the set of class into y and transforms to a python list
labelNames = y.unique().tolist() 

# Replace the string class labels into a int class labels 
y = y.transform(lambda x: labelNames.index(x))
