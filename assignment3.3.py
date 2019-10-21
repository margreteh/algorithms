import numpy as np
import math

print("Assignment 3.3: Decoupled Power Flow with R = X")


# Choose method
primal = 0
dual = 0
standard = 1

if primal:
    print("\nPrimal method:")
elif dual:
    print("\nDual method:")
elif standard:
    print("\nStandard method:")

buses = np.array([1, 2, 3])

v = np.ones(3)
theta = np.zeros(3)

# Network values
X12 = 0.2
X13 = 0.1
X23 = 0.15

# Impedances
Z12 = complex(X12, X12)
Z13 = complex(X13, X13)
Z23 = complex(X23, X23)

y12 = 1 / Z12
y13 = 1 / Z13
y23 = 1 / Z23

Y_not_bus = np.array([[(y12 + y13), y12, y13], [y12, (y12 + y23), y23], [y13, y23, (y13 + y23)]])

G = Y_not_bus.real
B = Y_not_bus.imag

# Equivalent impedances for R=0
Z12_eq = complex(0, X12)
Z13_eq = complex(0, X13)
Z23_eq = complex(0, X23)

y12_eq = 1 / Z12_eq
y13_eq = 1 / Z13_eq
y23_eq = 1 / Z23_eq

Y_not_bus_eq = np.array([[(y12_eq + y13_eq), y12_eq, y13_eq], [y12_eq, (y12_eq + y23_eq), y23_eq],
                         [y13_eq, y23_eq, (y13_eq + y23_eq)]])

G_eq = Y_not_bus_eq.real
B_eq = Y_not_bus_eq.imag

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
T_eq = np.zeros((3, 3))
U_eq = np.zeros((3, 3))

# Jacobian matrix
H = np.zeros((2, 2))  # Upper left
L = np.zeros((2, 2))  # Lower right
H_eq = np.zeros((2, 2))  # ser bort fra resistans
L_eq = np.zeros((2, 2))  # ser bort fra resistans

# Calculated power vectors
P_cal = np.zeros(3)
Q_cal = np.zeros(3)

# Delta vectors
delta_theta = np.zeros((2, 1))
delta_v = np.zeros((2, 1))

delta_P = np.zeros((2, 1))
delta_Q = np.zeros((2, 1))

# Defining error and max mismiatch
max_mismatch = 1
error = 0.001

it = 0

while max_mismatch > error:
    print("\nIteration nr.:", it + 1)

    # Calculating T and U
    for i in range(buses.size):
        for j in range(buses.size):
            T[i][j] = G[i][j] * math.cos(theta[i] - theta[j]) + B[i][j] * math.sin(theta[i] - theta[j])
            U[i][j] = G[i][j] * math.sin(theta[i] - theta[j]) - B[i][j] * math.cos(theta[i] - theta[j])

            T_eq[i][j] = G_eq[i][j] * math.cos(theta[i] - theta[j]) + B_eq[i][j] * math.sin(theta[i] - theta[j])
            U_eq[i][j] = G_eq[i][j] * math.sin(theta[i] - theta[j]) - B_eq[i][j] * math.cos(theta[i] - theta[j])

    # Calculating powers
    for i in range(buses.size):
        P_cal[i] = v[i] * v[i] * G[i][i]
        Q_cal[i] = -v[i] * v[i] * B[i][i]
        for j in range(buses.size):
            if i != j:
                P_cal[i] = P_cal[i] - v[i] * v[j] * T[i][j]
                Q_cal[i] = Q_cal[i] - v[i] * v[j] * U[i][j]

    print("\nCalculated active power:\n", P_cal)
    print("\nCalculated reactive power:\n", Q_cal)

    # Calculating the Jacobian matrix
    for i in range(2):
        for j in range(2):
            if i == j:
                H[i][j] = 0
                L[i][j] = -2 * v[i] * B[i][i]
                H_eq[i][j] = 0
                L_eq[i][j] = -2 * v[i] * B_eq[i][i]
                for k in range(buses.size):
                    if k != i:
                        H[i][j] = H[i][j] + v[i] * v[k] * U[i][k]
                        L[i][j] = L[i][j] - v[k] * U[i][k]
                        H_eq[i][j] = H_eq[i][j] + v[i] * v[k] * U_eq[i][k]
                        L_eq[i][j] = L_eq[i][j] - v[k] * U_eq[i][k]
            else:
                H[i][j] = -v[i] * v[j] * U[i][j]
                L[i][j] = - v[i] * U[i][j]
                H_eq[i][j] = -v[i] * v[j] * U_eq[i][j]
                L_eq[i][j] = - v[i] * U_eq[i][j]

    print("\nH:\n", H)
    print("\nL:\n", L)

    print("\nH_eq:\n", H_eq)
    print("\nL_eq:\n", L_eq)

    # Calculating mismatch
    for i in range(2):
        delta_P[i] = P_spes[i] - P_cal[i]
        delta_Q[i] = Q_spes[i] - Q_cal[i]

    print("\nMismatch:\nDeltaP:\n", delta_P, "\nDeltaQ:\n", delta_Q)

    if primal:
        # Primal method
        delta_theta = np.dot(np.linalg.inv(H), delta_P)
        delta_v = np.dot(np.linalg.inv(L_eq), delta_Q)

    elif dual:
        # Dual method
        delta_v = np.dot(np.linalg.inv(L), delta_Q)
        delta_theta = np.dot(np.linalg.inv(H_eq), delta_P)

    elif standard:
        # Standard method
        delta_theta = np.dot(np.linalg.inv(H_eq), delta_P)
        delta_v = np.dot(np.linalg.inv(L_eq), delta_Q)

    for i in range(2):
        theta[i] += delta_theta[i]
        v[i] += delta_v[i]

    print("\nCorrection:\nDeltaV:\n", delta_v, "\nDeltaTheta:\n", delta_theta)
    print("\nVoltage magnitudes:\n", v, "\nVoltage angles:\n", theta)

    # Calculating maximum mismatch
    max_P = max(abs(delta_P))
    max_Q = max(abs(delta_Q))
    max_mismatch = max(max_P, max_Q)

    # Updating iteration count
    it += 1
