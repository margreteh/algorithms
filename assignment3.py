import numpy as np
import math

print("Assignment 3: Decoupled Power Flow")

buses = np.array([1, 2, 3])
v = np.ones(3)
flat_start = 1

v = np.ones(3)
theta = np.zeros(3)

if not flat_start:
    v[0] = 0.83533749
    v[1] = 0.86371646
    theta[0] = -0.15291389
    theta[1] = -0.0949069

# Network values
R12 = 0.05
R13 = 0.05
R23 = 0.05

X12 = 0.2
X13 = 0.1
X23 = 0.15

# Impedances
Z12 = complex(R12, X12)
Z13 = complex(R13, X13)
Z23 = complex(R23, X23)

y12 = 1/Z12
y13 = 1/Z13
y23 = 1/Z23

Y_not_bus = np.array([[(y12+y13), y12, y13], [y12, (y12+y23), y23], [y13, y23, (y13+y23)]])

G = Y_not_bus.real
B = Y_not_bus.imag

# Load values
P1 = -1.0
P2 = -0.5
Q1 = -0.5
Q2 = -0.5

P_spes = ([P1, P2])
Q_spes = ([Q1, Q2])

# Arrays needed for calculations
T = np.zeros((3, 3))
U = np.zeros((3, 3))

# Jacobian matrix
H = np.zeros((2, 2))  # Upper left
L = np.zeros((2, 2))  # Lower right

# Calculated power vectors
P_cal = np.zeros(3)
Q_cal = np.zeros(3)

# Updating T and U
for i in range(buses.size):
    for j in range(buses.size):
        T[i][j] = G[i][j] * math.cos(theta[i] - theta[j]) + B[i][j] * math.sin(theta[i] - theta[j])
        U[i][j] = G[i][j] * math.sin(theta[i] - theta[j]) - B[i][j] * math.cos(theta[i] - theta[j])

# Calculating powers
for i in range(buses.size):
    P_cal[i] = v[i] * v[i] * G[i][i]
    Q_cal[i] = -v[i] * v[i] * B[i][i]
    for j in range(buses.size):
        if i != j:
            P_cal[i] = P_cal[i] - v[i] * v[j] * T[i][j]
            Q_cal[i] = Q_cal[i] - v[i] * v[j] * U[i][j]

# Calculating the Jacobian matrix
for i in range(2):
    for j in range(2):
        if i == j:
            H[i][j] = 0
            L[i][j] = -2 * v[i] * B[i][i]
            for k in range(buses.size):
                if k != i:
                    H[i][j] = H[i][j] + v[i] * v[k] * U[i][k]
                    L[i][j] = L[i][j] - v[k] * U[i][k]
        else:
            H[i][j] = -v[i] * v[j] * U[i][j]
            L[i][j] = - v[i] * U[i][j]



mismatch = np.zeros((4, 1))

max_mismatch = 1

error = 0.001

it = 1


