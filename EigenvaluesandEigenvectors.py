"""
Jared Jacobowitz
ESC251 Spring 2020
Workshop 3: Eigen Vectors and Eigen Values
"""

import matplotlib.pyplot as plt
import numpy as np

# Define A
A = np.array([[1, 1], [4, -2]])

# Initial conditions
IC = [2,-3]

# Solve eigenproblem
evals, evecs = np.linalg.eig(A)

v1 = evecs[:,0].reshape(len(evecs[0]),1)
v2 = evecs[:,1].reshape(len(evecs[1]),1)

s1 = evals[0]
s2 = evals[1]

# # Optional prints to check eigenproblem solution
# Av1 = np.dot(A, v1)
# Av2 = np.dot(A, v2)
# s1v1 = s1*v1
# s2v2 = s2*v2

# print('A * v1 =', Av1)
# print('s * v1 =', s1v1)
# print('A * v2 =', Av2)
# print('s * v2 =', s2v2)


# Solve for constants
x0 = np.array([IC[0], IC[1]])

# see https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.solve.html
c = np.linalg.solve(evecs, x0)

c1 = c[0]
c2 = c[1]

# Create solution
t = np.linspace(0, 10, 500)


# Solution
x = c1 * np.exp(s1*t) * v1 +  c2 * np.exp(s2*t) * v2

x1 = x[0]
x2 = x[1]

# Solution plot
plt.figure()
plt.plot(t, x1, label = r'$x_1$')
plt.plot(t, x2, label = r'$x_2$')

plt.xlim(-0.1, 1.5)
plt.ylim(-5, 20)
plt.xlabel('Time [s]')

plt.ylabel(r'$x_{1}$ or $x_{2}$')
plt.legend()
plt.title('Solution of IVP')

plt.savefig('EigenproblemSolution.png', dpi = 1000)

plt.show()

# Phase portrait
plt.figure()
window = 10

xvals = np.linspace(-window,window, 100)
yvals = np.linspace(-window,window, 100)

xvals, yvals = np.meshgrid(xvals, yvals)

xdot = A[0,0] * xvals + A[0,1] * yvals
ydot = A[1,0] * xvals + A[1,1] * yvals

plt.streamplot(xvals, yvals, xdot, ydot, density = 0.8, 
               linewidth = 0.3, color = 'black')

plt.plot(v1[0]*t, v1[1]*t, label = 'Eigen Vectors', color = 'black')
plt.plot(v2[0]*t, v2[1]*t, color = 'black')

plt.axvline(x=0, color='black', alpha = 0.2)
plt.axhline(y=0, color='black', alpha = 0.2)

plt.plot(IC[0], IC[1], 'o', color='#1f77b4', label = 'Initial Condition')
plt.plot(x1, x2, label = 'Path of IC')
plt.plot(IC[0], IC[1], 'o', color='#1f77b4') # plotted again to fix overlap


plt.xlim([-window,window])
plt.ylim([-window,window])
plt.xlabel(r'$x_1$')

plt.ylabel(r'$x_2$')
plt.legend()
plt.title('Phase Portrait with an Intial Condition Response')


plt.savefig('EigenproblemPhasePortrait.png', dpi = 800)

plt.show()