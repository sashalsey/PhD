from firedrake.__future__ import interpolate # type: ignore
import firedrake as fd # type: ignore
from firedrake.output import VTKFile # type: ignore
import firedrake.adjoint as fda # type: ignore
from pyMMAopt import MMASolver # type: ignore
import time
start = time.time()
fda.continue_annotation()

# 0 Basic Parameters
T = fd.Constant((0, -1000, 0)) # Traction Force
initial_rho = 0.1 # Initial density for each element
targ_V = 0.3 # Volume fraction constraint
max_iter = 25
E, nu = 209*10**9, 0.3                      # Young Modulus
mu = E/(2.0*(1.0 + nu))                     # Shear Modulus
p = fd.Constant(3)  # Penalisation parameter
eps = 0.0001 # Small value close to zero to prevent sigularity
'''print(T)
print(initial_rho)
print(max_iter)
print(targ_V)
print(E)
print(nu)
print(p)'''

# 1 Define Mesh (3D)
nelx, nely, nelz = 20, 10, 10
lx, ly, lz = 1, 1, 1
Vol = lx*ly*lz
mesh = fd.BoxMesh(nelx, nely, nelz, lx, ly, lz, hexahedral=True)
vtkfile = VTKFile("mesh.pvd")
vtkfile.write(mesh)
#print(nelx,nely,nelz)
# 2 Define Function Spaces
V = fd.VectorFunctionSpace(mesh, "CG", 1)  # Displacement space
R = fd.FunctionSpace(mesh, "DG", 1)  # Density space
uh = fd.Function(V) # Function to Hold Solution

# 3 Define Symbolic Variables
u = fd.TrialFunction(V)    # Displacement trial function
v = fd.TestFunction(V)     # Displacement test function
rho = fd.Function(R, name="Density")  # Density field as control variable
rho.interpolate(fd.Constant(initial_rho))
x, y, z = fd.SpatialCoordinate(mesh)

# 4 Define Boundary Conditions and Loads
''' * 1: plane x == 0
    * 2: plane x == Lx
    * 3: plane y == 0
    * 4: plane y == Ly
    * 5: plane z == 0
    * 6: plane z == Lz '''
bcs = fd.DirichletBC(V, fd.Constant((0, 0, 0)), 1)  # Zero displacement on left boundary

# 5 Define Stress and Strain
lambda_ = E*nu/((1.0 + nu)*(1.0 -2.0*nu))     # Lambda
Id = fd.Identity(mesh.geometric_dimension())
def epsilon(u): return 0.5 * (fd.grad(u) + fd.grad(u).T)
def sigma(u): return lambda_ * fd.div(u) * Id + 2 * mu * epsilon(u)
def simp(rho,p): return eps + (1 - eps)*rho**p
def von_mises_proj(uh):
    DG0 = fd.FunctionSpace(mesh, "DG", 0)

    sigma_xx = fd.project(sigma(uh)[0, 0], DG0)
    sigma_yy = fd.project(sigma(uh)[1, 1], DG0)
    sigma_zz = fd.project(sigma(uh)[2, 2], DG0)
    sigma_xy = fd.project(sigma(uh)[0, 1], DG0)
    sigma_yz = fd.project(sigma(uh)[1, 2], DG0)
    sigma_zx = fd.project(sigma(uh)[2, 0], DG0)

    von_mises_stress = fd.sqrt(0.5 * ((sigma_xx - sigma_yy)**2 + (sigma_yy - sigma_zz)**2 + (sigma_zz - sigma_xx)**2 + 6 * (sigma_xy**2 + sigma_yz**2 + sigma_zx**2)))
    von_mises_proj = fd.project(von_mises_stress, DG0)
    return von_mises_proj

# 6 Define Bilinear Form (Left-Hand Side) and Linear Form (Right-Hand Side)
a = simp(rho,p) * fd.inner(sigma(u), epsilon(v)) * fd.dx  # Weak form of elasticity equation
L = fd.dot(T, v) * fd.ds(2) # Linear form (load term) do on ds() boundary in 2D

# 7 Solve Variational Problem for Given Density Distribution (Compliance Formulation)
fd.solve(a == L, uh, bcs, solver_parameters={"mat_type": "aij",
                                             "ksp_type": "cg",
                                             "pc_type": "hypre",
                                             "pc_factor_shift_type": "inblocks"})

# 8 Define Compliance Objective for Optimization
J = fd.assemble(fd.dot(T, uh)*fd.ds(2)) # Objective: strain energy
m = fda.Control(rho) # Control variable

# 9 Define Reduced Functional
callback_data = {
    "pre_controls": None,
    "post_checkpoint": None,
    "post_derivatives": None,
    "post_values": None,}
def pre_callback(controls):
    #print("Pre-derivative callback triggered.")
    callback_data["pre_controls"] = controls
    return controls
iteration_derivatives_post = []
iteration_dJdrho = []
def post_callback(checkpoint, derivatives, values):
    #print("Post-derivative callback triggered.")
    callback_data["post_checkpoint"] = checkpoint 
    callback_data["post_derivatives"] = derivatives 
    callback_data["post_values"] = values 
    derivative_values = derivatives[0].dat.data[:]
    iteration_derivatives_post.append(derivative_values)
    dJdrho = fda.compute_gradient(J, m)
    dJdrho_plot = fd.Function(R)
    dJdrho_plot.assign(dJdrho)
    vtkfile = VTKFile("dJdrho.pvd")
    vtkfile.write(dJdrho_plot)
    return derivatives 

Jhat = fda.ReducedFunctional(J,m,derivative_cb_pre=pre_callback,derivative_cb_post=post_callback)

# 10 Define Contraints and Constrained Optimisation Problem
class VolumeConstraint(fda.InequalityConstraint):
        def __init__(self, Vlimit):
            self.Vlimit = float(Vlimit)
            self.tmp = fd.Function(R)
        
        def function(self, m):
            self.tmp.interpolate(m)
            integral = fd.assemble(1/Vol*self.tmp*fd.dx)
            print('Constraint is :', integral)
            value = self.Vlimit-integral
            return [value]

        def jacobian(self, m):
            vgradients = fd.assemble(fd.derivative(1/Vol*self.tmp*fd.dx, self.tmp))
            with vgradients.dat.vec as v:
                v.scale(-1.0)
            return [vgradients]

        def output_workspace(self):
            return [0.0]

        def length(self):
            """Return the number of components in the constraint vector (here, one)."""
            return 1

problem = fda.MinimizationProblem(Jhat, bounds=(0.001,1), constraints=[VolumeConstraint(targ_V)],)

# 11 Optimise Density Distribution to Minimise Weight
parameters_mma = {
        "move": 0.2,
        "maximum_iterations": max_iter,
        "m": 1,
        "IP": 0,
        "tol": 1e-3,
        "accepted_tol": 1e-3,
        "norm": "L2",
        "gcmma": False,
    }

solver = MMASolver(problem, parameters=parameters_mma)
fda.get_working_tape().progress_bar = fd.ProgressBar
rho_opt = solver.solve()

# 12 Analyse solution
rho_plot = fd.Function(R)
rho_plot.assign(rho_opt['control'])
volci = 1/Vol*fd.assemble(rho_plot * fd.dx) 
print(f"Global density (volume fraction): {volci:.6f}")
vtkfile = VTKFile("rho_plot.pvd")
vtkfile.write(rho_plot) 

if volci < targ_V:
    print("Volume fraction constraint satisfied.")
else: print("Volume fraction constraint exceeded.")

von_mises_fin = von_mises_proj(uh)
vtkfile = VTKFile("von_mises_fin.pvd")
vtkfile.write(von_mises_fin)
 
tape = fda.get_working_tape()
tape.clear_tape()
end = time.time()
print("Total time ", end - start)
