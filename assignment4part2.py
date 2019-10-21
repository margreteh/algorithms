import numpy as np

# impedances
R = 0

X_12 = 0.2
X_13 = 0.1
X_23 = 0.25
X_34 = 0.25

# Scheduled powers
P1 = -1.25
P2 = -0.4
P3 = -0.5

P_spes = np.zeros((3, 1))
P_spes[0][0] = P1
P_spes[1][0] = P2
P_spes[2][0] = P3

# Contingency parameters
cont_from = 1
cont_to = 3
delta_h = -5

# Matrices
H = [[15, -5, -10], [-5, 9, -4], [-10, -4, 18]]

M = np.zeros((3, 1))
M_transpose = np.zeros((1, 3))

M[cont_from-1][0] = 1
M[cont_to-1][0] = -1
M_transpose[0][cont_from-1] = 1
M_transpose[0][cont_to-1] = -1

delta = np.dot(np.linalg.inv(H), P_spes)
print("\nOriginal voltage angles:\n", delta)

temp = np.dot(M_transpose, np.linalg.inv(H))

c = 1/(1/delta_h + np.dot(temp, M))
c = c[0]

print("\nc:", c)

temp = np.dot(np.linalg.inv(H), M)
temp = np.dot(temp, M_transpose)
temp = np.dot(temp, delta)
delta_delta = -c * temp

print("\nDelta delta:\n", delta_delta)

for i in range(3):
    delta[i] += + delta_delta[i]

print("\nPost contingency voltage angles:\n", delta)

P12 = (delta[1-1]-delta[2-1])/X_12
P13 = (delta[1-1]-delta[3-1])/X_13
P23 = (delta[2-1]-delta[3-1])/X_23
P34 = (delta[3-1]-0)/X_34

print("\nLine flows:\nP12:", P12, "\nP13:", P13, "\nP23:", P23, "\nP34:", P34)
