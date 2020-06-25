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
        self.weights1 = np.random.rand(9, 5)
        self.weights2 = np.random.rand(5, 1)

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

    # feedforward function implementation
    def feedforward(self):
        # feedforward to layer 1
        self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))

        # feedforward to layer 2
        self.layer2 = self.sigmoid(np.dot(self.layer1, self.weights2))

        return self.layer2

    def backprop(self, learning_rate):
        # get partial derivative for layer 2 weights
        d_weights2 = np.dot(self.layer1.T, 2*(self.y - self.output)
                            * self.sigmoid_derivative(self.output))

        # get partial derivative for layer 1 weights
        d_weights1 = np.dot(self.input.T, np.dot(2*(self.y - self.output) * self.sigmoid_derivative(self.output), self.weights2.T)
                            * self.sigmoid_derivative(self.layer1))

        # adjust weights
        self.weights1 += learning_rate * d_weights1
        self.weights2 += learning_rate * d_weights2

    def train(self, X, y, learning_rate):
        # set input and labels
        self.input = X
        self.y = y

        # feedforward and setoutput
        self.output = self.feedforward()

        # backpropagate
        self.backprop(learning_rate)

        # return training error
        return np.mean(np.square(y - np.round(self.output)))

    def test(self, X, y):
        # set input and labels
        self.input = X
        self.y = y

        # feedforward and setoutput
        self.output = self.feedforward()

        # print test results
        print("\nTesting Results\nError : " +
              str(np.mean(np.square(y - np.round(self.output)))) + "\n")


NN = NeuralNetwork()
learning_rate = 0.01

# train this network for 1000 iterations
for i in range(1, 1001):
    error = NN.train(training_x, training_y, learning_rate)

    # print error after every 10 iterations
    if i % 10 == 0 or i == 1:
        print("Iteration : " + str(i)+" | Error : " + str(error))


NN.test(testing_x, testing_y)
