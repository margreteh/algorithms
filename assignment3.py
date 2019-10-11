import numpy as np

print("Assignment 3: Decoupled Power Flow")

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
y34 = 1/Z34

Y_not_bus = np.array([[(y12+y13), y12, y13], [y12, (y12+y23), y23], [y13, y23, (y13+y23)]])

G = Y_not_bus.real
B = Y_not_bus.imag

# Load values
P1 = -0.8
P2 = -0.5
Q1 = -0.4
Q2 = -0.5
