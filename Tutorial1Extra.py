import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
train_data = pd.read_csv('insurance_dataset.csv', sep=',')

# train_data = pd.get_dummies(train_data)
# print(train_data.head())

# converting categorical values to numerical values
claenup_values = {"sex": {"male": 1, "female": 0},
                  "smoker": {"yes": 1, "no": 0}, "region": {"southwest": 1, "southeast": 2, "northwest": 3, "northeast": 4}}
train_data.replace(claenup_values, inplace=True)

# convert pandas array to numpy array(optional)
train_data = train_data.to_numpy()
print(train_data[:, 0])

# print(train_data.head())
# print(train_data[4])

# if we use pandas array -->
# X1 = train_data['age']
# X2 = train_data['sex']
# X3 = train_data['bmi']
# X4 = train_data['children']
# X5 = train_data['smoker']
# X6 = train_data['region']
# Y = train_data['charges']

X1 = train_data[:, 0]
X2 = train_data[:, 1]
X3 = train_data[:, 2]
X4 = train_data[:, 3]
X5 = train_data[:, 4]
X6 = train_data[:, 5]

Y = train_data[:, 6]

m1 = 0
m2 = 0
m3 = 0
m4 = 0
m5 = 0
c = 0

L = 0.0001
iterations = 1000

n = float(len(X1))

for i in range(iterations):
    Y_pred = m1 * X1 + m2 * X2 + m3 * X3 + m4 * X4 + m5 * X5 + c
    D_m1 = (-2 / n) * sum(X1 * (Y - Y_pred))
    D_m2 = (-2 / n) * sum(X2 * (Y - Y_pred))
    D_m3 = (-2 / n) * sum(X3 * (Y - Y_pred))
    D_m4 = (-2 / n) * sum(X4 * (Y - Y_pred))
    D_m5 = (-2 / n) * sum(X5 * (Y - Y_pred))
    D_c = (-2 / n) * sum(Y - Y_pred)

    m1 = m1 - L * D_m1
    m2 = m2 - L * D_m2
    m3 = m3 - L * D_m3
    m4 = m4 - L * D_m4
    m5 = m5 - L * D_m5

    c = c - L * D_c

print("Y_pred = ", Y_pred)
print("m1 = ", m1)
print("m2 = ", m2)
print("m3 = ", m3)
print("m4 = ", m4)
print("m5 = ", m5)
print("c = ", c)

