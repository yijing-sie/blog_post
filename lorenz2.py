# -*- coding: utf-8 -*-
"""
Created on Wed May 12 21:10:25 2021

@author: jenny
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



#one of the critical points
fx, fy, fz = -np.sqrt(72),-np.sqrt(72),27

# Maximum time point and total number of time points.
N = 10000
step = N/100 #time step size = 0.01
tspan = np.linspace(0, step, N)

def lorenz(t, initial):
    """The Lorenz equations."""
    x, y, z = initial
    dxdt = 10*(y - x)
    dydt = x*(28 - z) - y
    dzdt = x*y - (8/3)*z
    return dxdt, dydt, dzdt

# Lorenz initial points.
initial = (1, 1, 1)
# Integrate the Lorenz equations.
soln = solve_ivp(lorenz, (tspan[0], tspan[-1]), initial, 
            dense_output=True)
# Interpolate solution onto the time grid, t.
x, y, z = soln.sol(tspan)

initial2 = (1.00001, 1.00001, 1.00001)
soln = solve_ivp(lorenz, (tspan[0], tspan[-1]), initial2, 
            dense_output=True)
# Interpolate solution onto the time grid, t.
x2, y2, z2 = soln.sol(tspan)


# Plot the Lorenz attractor using a Matplotlib 3D projection.
fig = plt.figure(facecolor='k', figsize=(10, 7.5))
ax = fig.gca(projection='3d')
ax.set_facecolor('w')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

title1 = 'initial point = ' + str(initial)  
title2 = 'initial point = ' + str(initial2)
ax.plot(x, y, z, color= 'red', linewidth = 0.1 , label = title1)
ax.plot(x2, y2, z2, color= 'green', linewidth = 0.1, label = title2)
ax.scatter(fx, fy, fz, c = 'blue')
ax.scatter(-fx, -fy, fz,  c = 'purple')
ax.legend()


# Remove all the axis clutter, leaving just the curve.
# ax.set_axis_off()


plt.savefig('lorenz.png', dpi=750)
plt.show()
