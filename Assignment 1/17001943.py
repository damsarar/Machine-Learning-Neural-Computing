# import numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt
import time
import random

# load the dataset
train_data = np.genfromtxt('dataset.csv', delimiter=',')
# remove the header row
train_data = np.delete(train_data, 0, 0)

# take SAT Score as X vector
X = train_data[:, 0]
print(X[:10])
# take GPA as Y vector
Y = train_data[:, 1]

# visualize data using matplotlib
plt.scatter(X, Y)
plt.xlabel('SAT Score')
plt.ylabel('GPA')
# plt.show()

#m = 0
#c = 0
m = random.random()
c = random.random()
L = 0.0001  # the learning rate
iterations = 1000  # the number os iterations to perform gradient descent

n = float(len(X))  # number of elements in X

# performing Gradient descent
for i in range(iterations):
    Y_pred = m*X + c  # the current predicted value of Y
    # derivative with respect to m
    D_m = (-1/n) * sum(X * (Y-Y_pred) / abs(Y-Y_pred))
    # derivative with respect to c
    D_c = (-1/n) * sum((Y-Y_pred) / abs(Y-Y_pred))
    m = m - L * D_m  # update m
    c = c - L * D_c  # update c

    # Final predictions
    Y_pred = m*X + c

    # Draw the best fitting line
    plt.scatter(X, Y)
    plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')
    plt.draw()
    plt.pause(1e-17)
    # time.sleep(0.1)
    plt.clf()

# output m and c
print("m = ", m)
print("c = ", c)

plt.show()
