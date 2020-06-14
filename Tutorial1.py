import numpy as np
import matplotlib.pyplot as plt

train_data = np.genfromtxt('dataset.csv', delimiter=',')
train_data = np.delete(train_data, 0, 0)

X = train_data[:, 0]
Y = train_data[:, 1]

# plt.scatter(X, Y)
# plt.xlabel('SAT Score')
# plt.ylabel('GPA')
# plt.show()

m = 0
c = 0

L = 0.0001
iterations = 1000

n = float(len(X))

for i in range(iterations):
    Y_pred = m * X + c
    D_m = (-2 / n) * sum(X * (Y - Y_pred))
    D_c = (-2 / n) * sum(Y - Y_pred)
    m = m - L * D_m
    c = c - L * D_c

print("m = ", Y_pred)
print("c = ", c)

Y_pred = m * X + c

plt.scatter(X, Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')
plt.show()
