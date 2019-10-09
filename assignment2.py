import numpy as np
import math

print("Assignment 2")

buses = np.array([1, 2, 3])
v = np.ones(3)
theta = np.zeros(3)

# Base case (these are for the values used in the slide, can find those for the assignment in assignment 1.2)
v[0] = 0.83533749
v[1] = 0.86371646
theta[0] = -0.15291389
theta[1] = -0.0949069

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

print("G:\n", G, "\nB:\n", B)

# Load values
P1 = -0.8
P2 = -0.5
Q1 = -0.4
Q2 = -0.5

Pact = np.array([P1, P2])
Qact = np.array([Q1, Q2])

# Arrays needed for calculations
T = np.zeros((3, 3))
U = np.zeros((3, 3))

# Jacobian matrix
J_1 = np.zeros((2, 2))  # Upper left
J_2 = np.zeros((2, 2))  # Upper right
J_3 = np.zeros((2, 2))  # Lower left
J_4 = np.zeros((2, 2))  # Lower right

# Calculated power vectors
Pcal = np.zeros(3)
Qcal = np.zeros(3)

# Mismatch vector
mismatch = np.zeros((5, 1))
mismatch[4][0] = 1

max_mismatch = 1

beta1 = 0.3
beta2 = 0.7
step_length = 0.3

J_ab = np.zeros((4, 1))
J_ab[0][0] = beta1
J_ab[1][0] = beta2

# -----------Predictor phase-----------

print("\nPredictor phase:")
# Jacobi for predictor phase
J_extra = np.array([0, 0, 0, 0, 1])

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
J_old = np.vstack((J_p, J_q))
J_temp = np.hstack((J_old, J_ab))
J = np.vstack((J_temp, J_extra))

print("\nJacobian matrix:\n", J)

print("\nMismatch vector:\n", mismatch)

# Calculating sensitivity vectors
sensitivity = np.linalg.solve(J, mismatch)

print("\nSensitivity vector:\n", sensitivity)

# Updating voltages and angles
for i in range(2):
    theta[i] = theta[i] + sensitivity[i][0]*step_length
    v[i] = v[i] + sensitivity[i+2][0]*step_length

print("\nVoltage magnitues:\n", v)
print("\nVoltage angles:\n", theta)

# ---------Corrector phase---------
J_extra = np.array([0, 0, 0, 0, 1])
print("\nCorrector phase:")

# Mismatch vector
mismatch = np.zeros((5, 1))

# Allowed error
error = 0.001

max_mismatch = 1

# Iteration count
it = 0

while max_mismatch > error and it < 4:
    print("\nIteration nr.", it+1)

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
    J_old = np.vstack((J_p, J_q))
    J_temp = np.hstack((J_old, J_ab))
    J = np.vstack((J_temp, J_extra))

    print("\nJacobian matrix:\n", J)

    # Updating mismatch vectors
    for i in range(2):
        mismatch[i][0] = Pact[i] - Pcal[i]
        mismatch[i + 2][0] = Qact[i] - Qcal[i]

    print("\nMismatch vector:\n", mismatch)

    # Calculating correction vectors
    corr = np.linalg.solve(J, mismatch)

    print("\nCorrection vector:\n", corr)

    # Updating voltages and angles
    for i in range(2):
        theta[i] = theta[i] + corr[i][0]
        v[i] = v[i] + corr[i + 2][0]

    print("\nVoltage magnitues:\n", v)
    print("\nVoltage angles:\n", theta)

    # Updating maximum correction
    max_mismatch = max(abs(mismatch))

    # Updating iteration count
    it += 1

# -----------Predictor phase-----------

print("\nPredictor phase 2:")
# Jacobi for predictor phase
J_extra = np.array([0, 0, 0, 0, 1])

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
J_old = np.vstack((J_p, J_q))
J_temp = np.hstack((J_old, J_ab))
J = np.vstack((J_temp, J_extra))

print("\nJacobian matrix:\n", J)

print("\nMismatch vector:\n", mismatch)

# Calculating sensitivity vectors
sensitivity = np.linalg.solve(J, mismatch)

print("\nSensitivity vector:\n", sensitivity)

# Updating voltages and angles
for i in range(2):
    theta[i] = theta[i] + sensitivity[i][0]*step_length
    v[i] = v[i] + sensitivity[i+2][0]*step_length

print("\nVoltage magnitues:\n", v)
print("\nVoltage angles:\n", theta)

# ---------Corrector phase---------
J_extra = np.array([0, 0, 0, 1, 0])
print("\nCorrector phase 2:")

# Mismatch vector
mismatch = np.zeros((5, 1))

# Allowed error
error = 0.001

max_mismatch = 1

# Iteration count
it = 0

while max_mismatch > error and it < 4:
    print("\nIteration nr.", it+1)

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
    J_old = np.vstack((J_p, J_q))
    J_temp = np.hstack((J_old, J_ab))
    J = np.vstack((J_temp, J_extra))

    print("\nJacobian matrix:\n", J)

    # Updating mismatch vectors
    for i in range(2):
        mismatch[i][0] = Pact[i] - Pcal[i]
        mismatch[i + 2][0] = Qact[i] - Qcal[i]

    print("\nMismatch vector:\n", mismatch)

    # Calculating correction vectors
    corr = np.linalg.solve(J, mismatch)

    print("\nCorrection vector:\n", corr)

    # Updating voltages and angles
    for i in range(2):
        theta[i] = theta[i] + corr[i][0]
        v[i] = v[i] + corr[i + 2][0]

    print("\nVoltage magnitues:\n", v)
    print("\nVoltage angles:\n", theta)

    # Updating maximum correction
    max_mismatch = max(abs(mismatch))

    # Updating iteration count
    it += 1

















