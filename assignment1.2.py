import numpy as np
import math
import matplotlib.pyplot as plt

print("Assignment 1, problem 2")

buses = np.array([1, 2, 3])
flat_start = 1

v = np.ones(3)
theta = np.zeros(3)

if not flat_start:
    v[0] = 0.8172113
    v[1] = 0.86018867
    theta[0] = -0.1432562
    theta[1] = -0.08088964

# Vectors for plotting
v_plot_bus_1 = []
v_plot_bus_2 = []
p_plot_bus_1 = []
p_plot_bus_2 = []
p_plot = []

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
P2 = -0.4
Q1 = -0.5
Q2 = -0.5

deltaP = -0.2

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
mismatch = np.zeros((4, 1))

# Defined allowed error
error = 0.001

for p in range(9):
    print("\nIncreasing P with ", deltaP * (p+1))
    Pact[0] = P1 + 0.3 * deltaP * (p + 1)
    Pact[1] = P2 + 0.7 * deltaP * (p + 1)

    # Setting mismatch to 1 before Newthon Raphson
    max_mismatch = 1

    # Iteration count for Newton Raphson
    it = 0

    while max_mismatch > error:
        # Updating iteration count
        it += 1

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

        # Updating voltages and angles
        for i in range(2):
            theta[i] = theta[i] + corr[i][0]
            v[i] = v[i] + corr[i+2][0]

        print("\nVoltage magnitues:\n", v)
        print("\nVoltage angles:\n", theta)

        # Finding biggest mismatch
        max_mismatch = max(abs(mismatch))

    # Adding to plotting vectors
    v_plot_bus_1.append(v[0])
    v_plot_bus_2.append(v[1])

    p_plot_bus_1.append(Pact[0])
    p_plot_bus_2.append(Pact[1])
    p_plot.append(-deltaP*(p+1))

print("Voltage bus 1: ", v_plot_bus_1)
print("Voltage bus 2: ", v_plot_bus_2)

# Plotting bus 1
plt.plot(p_plot, v_plot_bus_1)
plt.title("Voltage at bus 1 as a function of load at bus 1")
plt.ylabel("Voltage [pu]")
plt.xlabel("Load [pu]")
plt.show()

# Plotting bus 2
plt.plot(p_plot, v_plot_bus_2)
plt.title("Voltage at bus 2 as a function of load at bus 2")
plt.ylabel("Voltage [pu]")
plt.xlabel("Load [pu]")
plt.show()
