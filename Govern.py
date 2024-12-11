# Kumar Parameter determination and experimental validation of a wire feed additive manufacturing model
import firedrake as fd # type: ignore
import numpy as np # type: ignore
from firedrake.pyplot import plot # type: ignore
import matplotlib.pyplot as plt # type: ignore
from time import time 
import math as math

# Constants
Ts = 1873  # K, Solidus temperature
Tl = 1923  # K, Liquidus temperature
Tm = 1898  # K, Melting temperature
rho_1 = 4420.0  # kg/m^3, Density of solid Ti-6Al-4V Due to Boussinesq approx p1=p1s=p1l
mu_1 = 0.0035  # kg/(m.s), Dynamic viscosity of solid Ti-6Al-4V

# Domain and function spaces
mesh = fd.UnitSquareMesh(32, 32)
V = fd.VectorFunctionSpace(mesh, "CG", 1)  # Velocity space
Q = fd.FunctionSpace(mesh, "CG", 1)  # Scalar space (for temperature, pressure)

# Define fields
U = fd.Function(V, name="Velocity")  # Velocity field
P = fd.Function(Q, name="Pressure")  # Pressure field
T = fd.Function(Q, name="Temperature")  # Temperature field
alpha = fd.Function(Q, name="VOF")  # Volume of fluid (liquid fraction)

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

# Other constants
h_sf = 2.86e5  # J/kg, Enthalpy of solidification
beta = 8.0e-6  # 1/K, Thermal expansion coefficient
epsilon = lambda T: 0.1536 + 1.8377e-4 * (T - 298)  # Emissivity, varies with temperature
eta = 0.34  # Dynamic viscosity of liquid Ti-6Al-4V (unspecified conditions)

# Properties of argon gas
rho_2 = 1.6337 # kg/m^3
mu_2 = 42.26e-5 # kg/(m.s)
c_p2 = 520.0 # J/(kg.K)
k2 = 0.0177 # W/(m.K)

# 2 Laser Beam Profile
nu = 0.34 # Absorption coefficient
P_laser = 1e3 # W, Laser power
r_p = 0.65e-3 # m, Welding travel direction constant
r_t = 2.5e-3 # m, Welding transverse direction constant
U_laser = 5e-3 # m/s, Welding travel speed
def q_dotlaser(x,y,t): return 2*nu*P_laser / (np.pi*r_p*r_t)*np.exp(-2*((x-U_laser*t)/r_p)**2 - 2*(y/r_t)**2)

# 3.1 Governing Equations
alpha = 0.0 # Volume fraction of liquid metal {2}
def continuity_eqn(U): # Conservation of mass {3}
    return fd.div(U) == 0
def momentum_eqn(alpha, U, rho):# Conservation of momentum {4} incompressible fluid dp = 0
    return fd.div(rho * U) == 0 # dp/dt + ∇·(ρ*u) = 0
def surface_tension_term(U, sigma, kappa): # Surface tension term
    return fd.div(sigma * kappa * U)
def poisson_eqn(P, rho, U): # Poisson equation for pressure
    return fd.div(fd.grad(P)) == -fd.div(rho * U)
# Conservation of energy {5}
# Volumetric surface force {6}
n = np.div(alpha)/np.abs(np.div(alpha)) # {7}
k = -np.div(n) # {8}
C_D = 1e6
eD = 1e-3
def f_l(T):  return 0.5*math.erf(4*(T - Tm)/(Tl - Ts)) + 1 # Metal liquid fraction {10}
def S_D(T): return -C_D*(1 - f_l(T))**2 / (f_l(T)**3 + eD) * U # Darcy momentum sink term {9}
def Q_dotlaser(x,y,t,alpha,rho,rho_1,rho_2): return q_dotlaser(x,y,t)*np.abs(np.div(alpha))*2*rho/(rho_1+rho_2)
# 3.2 Material properties of one-fluid thermodynamic and transport mixture model
rho = alpha*rho_1 + (1 - alpha)*rho_2 # {12}
def mu(T): return alpha*f_l(T)*mu_1 + (1-alpha)*mu_2 # {13}
def cp(T): return alpha*(f_l(T)*c_p1(T)+(1-f_l(T))*c_p1(T))+(1-alpha)*c_p2 # {14}
def k(T): return alpha*(f_l(T)*k1(T)+(1-f_l(T))*k1(T))+(1-alpha)*k2 # {15}

# 3.3 Computational domain, boundary, and initial conditions
sigma_B = 5.67e-8 # W/m^2K^4, Stefan Boltzmann constant
T_a = 300 # K, Ambient temperature
def q_dotrad(T): return epsilon*sigma_B*(T**4 - T_a**4) # Radiation heat flux {16}