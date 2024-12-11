import firedrake as fd # type: ignore
import numpy as np # type: ignore
from firedrake.pyplot import plot # type: ignore
import matplotlib.pyplot as plt # type: ignore
from time import time 
import math as math

# Constants for material properties
Ts = 1873  # K, Solidus temperature
Tl = 1923  # K, Liquidus temperature
Tm = 1898  # K, Melting temperature
rho_1 = 4420.0  # kg/m^3, Density of solid Ti-6Al-4V (solid and liquid same due to Boussinesq approx)
mu_1 = 0.0035  # kg/(m.s), Dynamic viscosity of solid Ti-6Al-4V

# 1. Governing equations
# Defining the laser beam and simulation domain
alpha = fd.Function(fd.FunctionSpace(fd.UnitSquareMesh(32, 32), "CG", 1))  # VOF variable for liquid fraction
V = fd.VectorFunctionSpace(fd.UnitSquareMesh(32, 32), "CG", 1)  # Velocity space
U = fd.Function(V)  # Velocity field
P = fd.Function(fd.FunctionSpace(fd.UnitSquareMesh(32, 32), "CG", 1))  # Pressure field
T = fd.Function(fd.FunctionSpace(fd.UnitSquareMesh(32, 32), "CG", 1))  # Temperature field

# Heat capacities (J/(kg.K))
def c_p1(T):
    if 298 <= T <= 1268:
        return 483.04 + 0.215 * T
    elif 1268 < T <= Tm:
        return 412.7 + 0.1801 * T
    elif Tm <= T <= 3000:
        return 830.0
    else:
        return None  # Temperature outside the range

# 2. Momentum equation
# dp/dt + ∇·(ρ*u) = 0
def momentum_eqn(alpha, U, rho):
    return fd.div(rho * U) == 0

# Continuity equation for incompressible fluid flow
def continuity_eqn(U):
    return fd.div(U) == 0

# 3. Darcy sink term
C_D = 1e6  # Large constant for Darcy term
eD = 1e-3
def f_l(T):  return 0.5 * math.erf(4 * (T - Tm) / (Tl - Ts)) + 1  # Metal liquid fraction
def S_D(T): return -C_D * (1 - f_l(T)) ** 2 / (f_l(T) ** 3 + eD) * U  # Darcy momentum sink term {9}

# 4. Laser source term
P_laser = 1e3  # Laser power
r_p = 0.65e-3  # Beam radius in welding direction
r_t = 2.5e-3  # Beam radius in transverse direction
U_laser = 5e-3  # Welding travel speed
nu = 1  # Absorption coefficient

def q_dotlaser(x, y, t):
    return 2 * nu * P_laser / (np.pi * r_p * r_t) * np.exp(-2 * ((x - U_laser * t) / r_p) ** 2 - 2 * (y / r_t) ** 2)

# 5. Surface tension term
def surface_tension_term(U, sigma, kappa):
    return fd.div(sigma * kappa * U)

# 6. Poisson equation for pressure
def poisson_eqn(P, rho, U):
    return fd.div(fd.grad(P)) == -fd.div(rho * U)

# 7. Temperature-dependent variables
def rho(alpha, rho_1, rho_2):
    return alpha * rho_1 + (1 - alpha) * rho_2  # {12}
def mu(alpha, T):
    return alpha * f_l(T) * mu_1 + (1 - alpha) * mu_2  # {13}
def cp(alpha, T):
    return alpha * (f_l(T) * c_p1(T) + (1 - f_l(T)) * c_p1(T)) + (1 - alpha) * c_p2  # {14}
def k(alpha, T):
    return alpha * (f_l(T) * k1(T) + (1 - f_l(T)) * k1(T)) + (1 - alpha) * k2  # {15}

# 8. Radiation term
sigma_B = 5.67e-8  # Stefan-Boltzmann constant
T_a = 300  # K, Ambient temperature
epsilon = lambda T: 0.1536 + 1.8377e-4 * (T - 298)  # Emissivity

def q_dotrad(T):
    return epsilon(T) * sigma_B * (T ** 4 - T_a ** 4)  # Radiation heat flux {16}

# 9. Solving the equations (this is just the setup; you'd need to solve using Firedrake's solver)
# Setup problem: this part depends on your problem geometry, mesh, and boundary conditions
# Example:
# solve(momentum_eqn(alpha, U, rho), U)