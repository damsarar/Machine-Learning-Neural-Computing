import numpy as np

# importing the dataset
data = np.genfromtxt('breast-cancer-wisconsin.data', delimiter=',')

# numpy.isnan returns a boolean array which has the value True everywhere that 'data' is not-a-number
# logical-not operator, ~ to get an array with Trues everywhere that 'data' is a valid number
data = data[~np.isnan(data).any(axis=1)]

# removing the first column
# axis 0 is thus the first dimension (rows), and axis 1 is the second dimension (columns)
data = np.delete(data, 0, axis=1)

# replace class with 0 and 1
data[:, 9][data[:, 9] == 2] = 0
data[:, 9][data[:, 9] == 4] = 1

# shuffle data rows
np.random.shuffle(data)

# seperate data to 2 parts as attributes and labels
attributes, labels = data[:, :9], data[:, 9:]

# normalizing attributes
x_min, x_max = attributes.min(), attributes.max()
attributes = (attributes-x_min)/(x_max-x_min)

# dividing dataset into training and testing data
margin = len(data)//10*7
training_x, testing_x = attributes[:margin, :], attributes[margin:, :]
training_y, testing_y = labels[:margin, :], labels[margin:, :]


class NeuralNetwork:
    def __init__(self):
        # initialize weights with randow values
        self.weights1 = np.random.rand(9, 4)
        self.weights2 = np.random.rand(4, 1)

        # declare variables for pred_output, input and labels
        self.output = None
        self.input = None
        self.y = None

    # sigmoid function implementation
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    # derivative of sigmoid function implementation
    def sigmoid_derivative(self, z):
        return z*(1-z)
