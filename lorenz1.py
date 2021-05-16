import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

#one of the critical points
fx, fy, fz = -np.sqrt(72),-np.sqrt(72),27
# Lorenz paramters and initial points.
initial = (1, 1, 1)
# Maximum time point and total number of time points.
N = 10000
step = N/100 #time step size = 0.01
tspan = np.linspace(0, step, N)

#Lorenz equations
def lorenz(t, initial):
    x, y, z = initial
    dxdt = 10*(y - x)
    dydt = x*(28 - z) - y
    dzdt = x*y - (8/3)*z
    return dxdt, dydt, dzdt

# Integrate the Lorenz equations.
soln = solve_ivp(lorenz, (tspan[0], tspan[-1]), initial, 
            dense_output=True)
# Interpolate solution onto the time grid, t.
x, y, z = soln.sol(tspan)
def find_orbit(x,y):
    if math.dist([x,y], [fx, fy]) < math.dist([x,y], [-fx, -fy]) :
        return True
    else:
        return False
result = [find_orbit(x[i],y[i]) for i in range(N)]


# Plot the Lorenz attractor using a Matplotlib 3D projection.
fig = plt.figure(facecolor='k', figsize=(10, 7.5))
ax = fig.gca(projection='3d')
ax.set_facecolor('w')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

pt = np.vstack((x, y, z)).T
p0 = []
p1 = []
for i,j in enumerate(result):
    if j: p0 += [pt[i]]
    else : p1 += [pt[i]]
p0 = np.array(p0)
p1 = np.array(p1)
        
c = ['red','green']
ax.plot(p0[:,0], p0[:,1], p0[:,2], color= 'red', alpha=1, linewidth = 0.5)
ax.plot(p1[:,0], p1[:,1], p1[:,2], color= 'green', alpha=1, linewidth = 0.5)
c0 = ax.scatter(fx, fy, fz, c = 'blue')
c1 = ax.scatter(-fx, -fy, fz,  c = 'purple')
title = 'initial point = ' + str(initial)
plt.title(label= title, fontsize=40, color="white")


plt.savefig('lorenz.png', dpi=750)
plt.show()
