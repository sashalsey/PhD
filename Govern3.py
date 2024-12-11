# Huang 2021 Laser wire-feed metal additive manufacturing of the Al alloy
import firedrake as fd # type: ignore
import numpy as np # type: ignore
from firedrake.pyplot import plot # type: ignore
import matplotlib.pyplot as plt # type: ignore
from time import time 
import math as math

P = 1e3 # W, Laser power
A_k = 0.34 # Absorption coefficient
def Q_1(t): return P * A_k * t

T_a = 300 # K, Ambient temperature
m = 1 # kg, Mass of melted wire
T_s = 1873  # K, Solidus temperature
T_l = 1923  # K, Liquidus temperature
T_m = 1898  # K, Melting temperature
c_p2 = 520.0 # J/(kg.K)
def f_l(T):  return 0.5*math.erf(4*(T - Tm)/(Tl - Ts)) + 1 # Metal liquid fraction {10}
def c_p1(T):
    if 298 <= T <= 1268:
        return 483.04 + 0.215 * T
    elif 1268 < T <= Tm:
        return 412.7 + 0.1801 * T
    elif Tm <= T <= 3000:
        return 830.0
    else:
        return None  # Temperature outside the range
    
T_values = np.linspace(298,3000,100)
c_p_values = [c_p1(T) for T in T_values]
plt.plot(T_values, c_p_values)
plt.xlabel('Temperature (K)')
plt.ylabel('Specific Heat Capacity (J/kg.K)')
plt.grid(True)
plt.show()
dH_F = 286e3 # J/kg, Latent heat of fusion Ti
Q_2 = m*(c_p2*(T_m - T_a) + dH_F)