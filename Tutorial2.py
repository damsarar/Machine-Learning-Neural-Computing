import numpy as np

# importing the dataset
data = np.genfromtxt('breast-cancer-wisconsin.data', delimiter=',')

# numpy.isnan returns a boolean array which has the value True everywhere that 'data' is not-a-number
# logical-not operator, ~ to get an array with Trues everywhere that 'data' is a valid number
data = data[~np.isnan(data).any(axis=1)]

# removing the first column
# axis 0 is thus the first dimension (rows), and axis 1 is the second dimension (columns)
data = np.delete(data, 0, axis=1)
