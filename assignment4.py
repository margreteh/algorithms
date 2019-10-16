import numpy as np

print("Assignment 3: Decoupled Power Flow")

buses = np.array([1, 2, 3, 4])
v = np.ones(4)
theta = np.zeros(4)

# Network values
R12 = 0
R13 = 0
R23 = 0
R34 = 0

X12 = 0.2
X13 = 0.1
X23 = 0.25
X34 = 0.25

# Load values
P1 = -1.25
P2 = -0.4
P3 = -0.6

Bptdf = [[15, -5, -10], [-5, 9, -4], [-10, -4, 18]]
P = [P1, P2, P3]

# Solving the load flow
angles = np.linalg.solve(Bptdf, P)

for i in range(3):
    theta[i] = angles[i]

print("\nVoltage angles:\n", theta)

P12 = (theta[1-1]-theta[2-1])/X12
P13 = (theta[1-1]-theta[3-1])/X13
P23 = (theta[2-1]-theta[3-1])/X23
P34 = (theta[3-1]-theta[4-1])/X34

print("\nLine flows:\nP12:", P12, "\nP13:", P13, "\nP23:", P23, "\nP34:", P34)

# Calculating distribution factors

# Line 1-2
one_over_x12 = [1/X12, -1/X12, 0]

a12 = np.linalg.solve(Bptdf, one_over_x12)

print("\nDistribution factors for line 1-2:\n", a12)

# Line 1-3
one_over_x13 = [1/X13, 0, -1/X13]

a13 = np.linalg.solve(Bptdf, one_over_x13)

print("\nDistribution factors for line 1-3:\n", a13)

# Line 3-4
one_over_x34 = [0, 0, 1/X34]

a34 = np.linalg.solve(Bptdf, one_over_x34)

print("\nDistribution factors for line 1-2:\n", a34)

# Load increasing on bus 1
dP = -0.5
P1 += dP

print("\nIncreasing P1 with 0.5:\n")

Bptdf = [[15, -5, -10], [-5, 9, -4], [-10, -4, 18]]
P = [P1, P2, P3]

# Solving the load flow
angles = np.linalg.solve(Bptdf, P)

for i in range(3):
    theta[i] = angles[i]

print("\nVoltage angles:\n", theta)

P12 = (theta[1-1]-theta[2-1])/X12
P13 = (theta[1-1]-theta[3-1])/X13
P23 = (theta[2-1]-theta[3-1])/X23
P34 = (theta[3-1]-theta[4-1])/X34

print("\nLine flows:\nP12:", P12, "\nP13:", P13, "\nP23:", P23, "\nP34:", P34)

# Load increasing on bus 1 and decreasing load on bus 2
dP2 = 0.3
P2 += dP2

print("\nIncreasing P1 with 0.5 and decreasing P2 with 0.3:\n")

Bptdf = [[15, -5, -10], [-5, 9, -4], [-10, -4, 18]]
P = [P1, P2, P3]

# Solving the load flow
angles = np.linalg.solve(Bptdf, P)

for i in range(3):
    theta[i] = angles[i]

print("\nVoltage angles:\n", theta)

P12 = (theta[1-1]-theta[2-1])/X12
P13 = (theta[1-1]-theta[3-1])/X13
P23 = (theta[2-1]-theta[3-1])/X23
P34 = (theta[3-1]-theta[4-1])/X34

print("\nLine flows:\nP12:", P12, "\nP13:", P13, "\nP23:", P23, "\nP34:", P34)
