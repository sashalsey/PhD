import firedrake as fd  # type: ignore
import numpy as np  # type: ignore
from firedrake import inner, grad, div, dx, dot # type: ignore
import math as math
import matplotlib as plt # type: ignore

# Domain and function spaces
mesh = fd.UnitSquareMesh(32, 32)
V = fd.VectorFunctionSpace(mesh, "CG", 1)  # Velocity space
Q = fd.FunctionSpace(mesh, "CG", 1)  # Scalar space (for temperature, pressure)

# Define fields
U = fd.Function(V, name="Velocity")  # Velocity field
P = fd.Function(Q, name="Pressure")  # Pressure field
T = fd.Function(Q, name="Temperature")  # Temperature field
alpha = fd.Function(Q, name="VOF")  # Volume of fluid (liquid fraction)

# Constants and parameters
C_D = 1e6  # Darcy coefficient
eD = 1e-3  # Small constant to avoid singularity in Darcy term
Ts = 1873  # Solidus temperature (K)
Tl = 1923  # Liquidus temperature (K)
Tm = 1898  # Melting temperature (K)
rho_1 = 4420.0  # Density of solid Ti-6Al-4V (kg/m^3)
mu_1 = 0.0035  # Dynamic viscosity of solid Ti-6Al-4V (Pa.s)
rho_2 = 1.6337  # Density of argon gas (kg/m^3)
mu_2 = 42.26e-5  # Dynamic viscosity of argon gas (Pa.s)
P_laser = 1e3  # Laser power (W)
r_p = 0.65e-3  # Welding travel direction constant (m)
r_t = 2.5e-3  # Welding transverse direction constant (m)
U_laser = 5e-3  # Welding travel speed (m/s)
nu = 0.34 # Absorption coefficient
sigma_0 = 1.6 # N/m
d_sigma_dT = -2.6e-4 # N/(mK)

rho = alpha*rho_1 + (1 - alpha)*rho_2 # {12}

# Specific heat capacities (J/(kg.K))
def c_p1(T):
    if 298 <= T <= 1268:
        return 483.04 + 0.215 * T
    elif 1268 < T <= Tm:
        return 412.7 + 0.1801 * T
    elif Tm <= T <= 3000:
        return 830.0
    else:
        return None  # Temperature outside the range

# Thermal conductivity (W/(m.K))
def k1(T):
    if 298 <= T <= 1268:
        return 1.2595 + 0.0157 * T
    elif 1268 < T <= Tm:
        return 3.5127 + 0.0127 * T
    elif Tm <= T <= 3000:
        return -12.752 + 0.024 * T
    else:
        return None  # Temperature outside the range

# Surface tension (N/m) as a function of temperature
def sigma(T):
    if Tm <= T <= 3000:
        return sigma_0 - (d_sigma_dT * (T - Tm))
    else:
        return None
sigma_0 = None  # Initial surface tension (depends on the material)
d_sigma_dT = None  # Rate of change of surface tension with temperature

# Compute the interface normal vector n
n = fd.grad(alpha) / fd.sqrt(fd.inner(fd.grad(alpha), fd.grad(alpha)) + 1e-8)  # {7} Add a small term to avoid division by zero

# Compute curvature kappa {8}
kappa = -fd.div(n)

# Surface tension force per unit volume: F_sigma = sigma(T) * kappa * n
F_sigma = sigma(T) * kappa * n

# Liquid fraction function
def f_l(T):
    return 0.5 * math.erf(4 * (T - Tm) / (Tl - Ts)) + 1

def mu(T): return alpha*f_l(T)*mu_1 + (1-alpha)*mu_2 # {13}

# Laser heat source term
def q_dotlaser(x, y, t):
    return 2 * nu * P_laser / (np.pi * r_p * r_t) * np.exp(-2 * ((x - U_laser * t) / r_p)**2 - 2 * (y / r_t)**2)

# Continuity equation (mass conservation)
def continuity_eqn(U):
    return fd.div(U) * dx

# Momentum equation: add the surface tension force F_sigma
def momentum_eqn(U, rho, P, mu, F_sigma):
    return (
        fd.inner(rho * fd.grad(U) * U, U) * fd.dx  # Convective term
        + mu * fd.inner(fd.grad(U), fd.grad(U)) * fd.dx  # Viscous term
        - P * fd.div(U) * fd.dx  # Pressure term
        + fd.inner(F_sigma, U) * fd.dx  # Surface tension term
    )

# Define boundary conditions and initial conditions (example)
U.assign(0)  # Initial velocity
alpha.assign(0.5)  # Initial volume fraction

# Energy equation (with laser heat source)
def energy_eqn(T, k, cp, rho, q_dotlaser):
    return (
        rho * cp * (fd.inner(fd.grad(T), fd.grad(T)) * dx)
        - q_dotlaser * T * dx
    )

# Surface tension force
def surface_tension_term(U, sigma, kappa):
    return fd.div(sigma * kappa * U) * dx

# Poisson equation for pressure
def poisson_eqn(P, rho, U):
    return fd.div(fd.grad(P)) + fd.div(rho * U) * dx

# Boundary and initial conditions
T.assign(Ts)  # Initial temperature field
U.assign(0)  # Initial velocity field
alpha.assign(0.5)  # Initial VOF

# Define the variational problems for the fields
F_momentum = momentum_eqn(U, rho, P, mu, F_sigma)
F_energy = energy_eqn(T, k1(T), c_p1(T), rho_1, q_dotlaser)
F_continuity = continuity_eqn(U)

# Solve each equation in a time-stepping loop
t = 0.0
T_end = 10.0
dt = 0.01

while t < T_end:
    fd.solve(F_momentum == 0, U)
    fd.solve(F_energy == 0, T)
    fd.solve(F_continuity == 0, P)
    t += dt

# Plot the volume fraction (alpha)
fig, ax = plt.subplots()
plot(alpha, axes=ax)
ax.set_title("Volume Fraction (alpha)")
plt.colorbar(ax.collections[0], ax=ax, label="Alpha")
plt.show()

# Plot the temperature field (T)
fig, ax = plt.subplots()
plot(T, axes=ax)
ax.set_title("Temperature Field")
plt.colorbar(ax.collections[0], ax=ax, label="Temperature (K)")
plt.show()

# Plot the velocity field (U)
fig, ax = plt.subplots()
plot(U, axes=ax)
ax.set_title("Velocity Field")
plt.quiver(U.dat.data_ro[:,0], U.dat.data_ro[:,1])  # Example for velocity vectors
plt.show()