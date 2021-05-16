#!/afs/andrew.cmu.edu/usr18/ysie/condor_public/bin/python
import numpy as np
from scipy.integrate import solve_ivp
import os

x0, y0, z0 = 1, 1, 1
initial = (x0, y0, z0)+ np.random.random(3)*0.3

# Maximum time point and total number of time points.
tspan = np.linspace(0, 100, 10000)

def lorenz(t, initial):
    """The Lorenz equations."""
    x, y, z = initial
    dxdt = 10*(y - x)
    dydt = x*(28 - z) - y
    dzdt = x*y - (8/3)*z
    return dxdt, dydt, dzdt

# Integrate the Lorenz equations.
soln = solve_ivp(lorenz, (tspan[0], tspan[-1]), initial, dense_output=True)
# Interpolate solution onto the time grid, t.
x, y, z = soln.sol(tspan)


ID  = os.getpid()
np.savetxt(str(ID)+".csv", np.vstack((x, y, z)).T, delimiter=',') 