import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time

train_data = np.genfromtxt('dataset.csv', delimiter=',')
train_data = np.delete(train_data, 0, 0)

X = train_data[:, 0]
Y = train_data[:, 1]

m = random.random()
c = random.random()

L = 0.0001
iterations = 1000

n = float(len(X))

plt.show()
plt.scatter(X, Y)
plt.xlabel('SAT Score')
plt.ylabel('GPA')

axes = plt.gca()

Y_pred = m * X + c
line, = axes.plot(X, Y_pred, 'r-')

for i in range(iterations):

    D_m = (-1 / n) * sum((X * (Y - Y_pred))/abs((Y - Y_pred)))
    D_c = (-1 / n) * sum((Y - Y_pred)/abs(Y - Y_pred))
    m = m - L * D_m
    c = c - L * D_c
    Y_pred = m * X + c

    line.set_xdata([min(X), max(X)])
    line.set_ydata([min(Y_pred), max(Y_pred)])
    plt.draw()
    plt.pause(1e-17)


print("m = ", m)
print("c = ", c)

line.set_xdata([min(X), max(X)])
line.set_ydata([min(Y_pred), max(Y_pred)])
plt.draw()
