import numpy as np
import math

print("Assignment 1, problem 1")

buses = np.array([1, 2, 3])
v = np.ones(3)
theta = np.zeros(3)

# Network values
R12 = 0.1
R13 = 0.05
R23 = 0.05
X12 = 0.2
X13 = 0.25
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
P1 = -0.8
P2 = -0.5
Q1 = -0.4
Q2 = -0.5

Pact = ([P1, P2])
Qact = ([Q1, Q2])

# Arrays needed for calculations
T = np.zeros((3, 3))
U = np.zeros((3, 3))

# Jaconian matrix
J_1 = np.zeros((2, 2))  # Upper left
J_2 = np.zeros((2, 2))  # Upper right
J_3 = np.zeros((2, 2))  # Lower left
J_4 = np.zeros((2, 2))  # Lower right

# Calculated power vectors
Pcal = np.zeros(3)
Qcal = np.zeros(3)

# Mismatch vectors
# mismatch_p = np.zeros(2)
# mismatch_q = np.zeros(2)
mismatch = np.zeros((4, 1))

# mismatch_t = np.hstack((mismatch_p, mismatch_q))
# mismatch = np.transpose(mismatch_t)
#
max_mismatch = 1
#
# # Defined allowed error
error = 0.001

# Iteration count
it = 1

while max_mismatch > error and it < 4:
    print("\nIteration nr. ", it, ":")

    # Updating T and U
    for i in range(buses.size):
        for j in range(buses.size):
            T[i][j] = G[i][j] * math.cos(theta[i] - theta[j]) + B[i][j] * math.sin(theta[i] - theta[j])
            U[i][j] = G[i][j] * math.sin(theta[i] - theta[j]) - B[i][j] * math.cos(theta[i] - theta[j])

    # Calculating powers
    for i in range(buses.size):
        Pcal[i] = v[i] * v[i] * G[i][i]
        Qcal[i] = -v[i] * v[i] * B[i][i]
        for j in range(buses.size):
            if i != j:
                Pcal[i] = Pcal[i] - v[i] * v[j] * T[i][j]
                Qcal[i] = Qcal[i] - v[i] * v[j] * U[i][j]

    # Calculating the Jacobian matrix
    for i in range(2):
        for j in range(2):
            if i == j:
                J_1[i][j] = 0
                J_2[i][j] = 2 * v[i] * G[i][i]
                J_3[i][j] = 0
                J_4[i][j] = -2 * v[i] * B[i][i]
                for k in range(buses.size):
                    if k != i:
                        J_1[i][j] = J_1[i][j] + v[i] * v[k] * U[i][k]
                        J_2[i][j] = J_2[i][j] - v[k] * T[i][k]
                        J_3[i][j] = J_3[i][j] - v[i] * v[k] * T[i][k]
                        J_4[i][j] = J_4[i][j] - v[k] * U[i][k]
            else:
                J_1[i][j] = -v[i] * v[j] * U[i][j]
                J_2[i][j] = -v[i] * T[i][j]
                J_3[i][j] = v[i] * v[j] * T[i][j]
                J_4[i][j] = - v[i] * U[i][j]

    print("\nCalculated active power:\n", Pcal)
    print("\nCalculated reactive power:\n", Qcal)

    # Merging
    J_p = np.hstack((J_1, J_2))
    J_q = np.hstack((J_3, J_4))
    J = np.vstack((J_p, J_q))

    print("\nJacobian matrix:\n", J)

    # Updating mismatch vectors
    for i in range(2):
        mismatch[i][0] = Pact[i] - Pcal[i]
        mismatch[i+2][0] = Qact[i] - Qcal[i]

    print("\nMismatch vector:\n", mismatch)

    # Calculating correction vectors
    corr = np.linalg.solve(J, mismatch)

    print("\nCorrection vector:\n", corr)

    for i in range(2):
        theta[i] = theta[i] + corr[i][0]
        v[i] = v[i] + corr[i+2][0]

    print("\nVoltage magnitues:\n", v)
    print("\nVoltage angles:\n", theta)

    # Finding biggest mismatch
    max_mismatch = max(abs(mismatch))

    # Updating iteration count
    it += 1















