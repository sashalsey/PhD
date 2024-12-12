#!/usr/bin/env python3
"""
Created on Fri Dec 15 12:07:58 2023

@author: rdm4317
"""
import numpy as np  # type: ignore
import firedrake as fd # type: ignore
from firedrake.output import VTKFile # type: ignore
import firedrake.adjoint as fda # type: ignore

fda.continue_annotation()

###############################################################################
class ForwardSolve:

    ###############################################################################
    def __init__(self, outputFolder, beta, penalisationExponent, variableInitialisation, rho0):

        # append inputs to class
        self.outputFolder = outputFolder

        # rho initialisation
        self.variableInitialisation = variableInitialisation
        self.penalisationExponent = penalisationExponent
        self.rho0 = rho0

        # material properties
        self.E0 = 1e-6  # [Pa] # void
        self.E1 = 1e0  # [Pa] # dense
        self.nu = 0.3  # []

        # pseudo-density functional
        self.beta = beta  # heaviside projection parameter
        self.eta0 = 0.5  # midpoint of projection filter

    ###############################################################################
    def GenerateMesh(self,):
        self.mesh = fd.BoxMesh(self.nx, self.ny, self.nz, self.lx, self.ly, self.lz, hexahedral=False)
        self.gradientScale = (self.nx * self.ny * self.nz) / (self.lx * self.ly * self.lz)  # firedrake bug?

    ###############################################################################
    def Setup(self):

        # mesh, functionals and associated static parameters
        ###############################################################################
        # generate mesh
        self.nx, self.ny, self.nz = 60, 20, 20
        self.lx, self.ly, self.lz = 0.3, 0.1, 0.1
        self.GenerateMesh()

        # compute mesh volume
        self.meshVolume = fd.assemble(1 * fd.dx(domain=self.mesh))

        # define function spaces
        self.functionSpace = fd.FunctionSpace(self.mesh, "CG", 1)
        self.vectorFunctionSpace = fd.VectorFunctionSpace(self.mesh, "CG", 1)

        # define psuedo-density function
        self.rho = fd.Function(self.functionSpace)

        # number of nodes in mesh
        self.numberOfNodes = len(self.rho.vector().get_local())

        # compute helmholtz filter radius - calc average element length
        self.helmholtzFilterRadius = 1 * (self.meshVolume / self.mesh.num_cells()) ** (1 / self.mesh.cell_dimension())

        # boundary conditions
        # add new boundary conditions by adding new key to dict
        bcDict = {}
        bcDict[0] = fd.DirichletBC(self.vectorFunctionSpace, fd.Constant((0, 0, 0)), 1)
        self.bcs = [bcDict[i] for i in range(len(bcDict.keys()))]

        # define output files
        # projected psuedo-density output file
        self.rho_hatFile = VTKFile(self.outputFolder + "rho_hat.pvd")
        self.rho_hatFunction = fd.Function(self.functionSpace, name="rho_hat")

        self.uFile = VTKFile(self.outputFolder + "u.pvd")
        self.uFunction = fd.Function(self.vectorFunctionSpace, name="u")

    ###############################################################################
    def ComputeInitialSolution(self):

        if self.variableInitialisation is True:
            initialisationFunction = fd.Function(self.functionSpace)
            initialisationFunction.vector().set_local(self.rho0)
        else:
            # initialise function for initial solution
            initialisationFunction = fd.Function(self.functionSpace)

            # generate uniform initialisation
            initialisationFunction.assign(0.3)

        return initialisationFunction.vector().get_local()

    ###############################################################################
    def CacheDesignVariables(self, designVariables, initialise=False):

        if initialise is True:
            # initialise with zero length array
            self.rho_np_previous = np.zeros(0)
            cache = False
        else:
            # determine whether the current design variables were simulated at previous iteration
            if np.array_equal(designVariables, self.rho_np_previous):

                # update cache boolean
                cache = True

            else:
                # new array is unique
                # assign current array to cache
                self.rho_np_previous = designVariables

                # update self.rho
                self.rho.vector().set_local(designVariables)

                # update cache boolean
                cache = False

        return cache

    ###############################################################################
    def Solve(self, designVariables):

        # insert and cache design variables
        # automatically updates self.rho
        identicalVariables = self.CacheDesignVariables(designVariables) # have they already been calcd

        if identicalVariables is False:

            # Helmholtz Filter
            ###############################################################################
            # trial and test functions
            u = fd.TrialFunction(self.functionSpace)
            v = fd.TestFunction(self.functionSpace)

            # DG specific relations
            n = fd.FacetNormal(self.mesh)
            h = 2 * fd.CellDiameter(self.mesh)
            h_avg = (h("+") + h("-")) / 2
            alpha = 1

            # weak variational form
            a = (fd.Constant(self.helmholtzFilterRadius) ** 2) * ( # this is for DG (jump)
                fd.dot(fd.grad(v), fd.grad(u)) * fd.dx
                - fd.dot(fd.avg(fd.grad(v)), fd.jump(u, n)) * fd.dS
                - fd.dot(fd.jump(v, n), fd.avg(fd.grad(u))) * fd.dS
                + alpha / h_avg * fd.dot(fd.jump(v, n), fd.jump(u, n)) * fd.dS
            ) + fd.inner(u, v) * fd.dx
            L = fd.inner(self.rho, v) * fd.dx

            # solve helmholtz equation
            self.rho_ = fd.Function(self.functionSpace, name="rho_")
            fd.solve(a == L, self.rho_)

            # projection filter - makes sure they're more b&w, sigmoid fn, beta = inf would be step fn, increase beta in continuation
            ###############################################################################
            self.rho_hat = (
                fd.tanh(self.beta * self.eta0) + fd.tanh(self.beta * (self.rho_ - self.eta0))
            ) / (fd.tanh(self.beta * self.eta0) + fd.tanh(self.beta * (1 - self.eta0)))

            # output rho_hat visualisation
            self.rho_hatFunction.assign(fd.project(self.rho_hat, self.functionSpace))
            self.rho_hatFile.write(self.rho_hatFunction)

            # linear elasticity load case
            ###############################################################################

            # define trial and test functions
            u = fd.TrialFunction(self.vectorFunctionSpace)
            v = fd.TestFunction(self.vectorFunctionSpace)

            # define surface traction
            x, y, z = fd.SpatialCoordinate(self.mesh)
            T = fd.conditional(fd.gt(x, 0.3 - (0.3 / 360) - 1e-8),
                    fd.conditional(fd.gt(y, 0.05 - 3 * (0.1 / 120) - 1e-8),
                    fd.conditional(
                        fd.lt(y, 0.05 + 3 * (0.1 / 120) + 1e-8),
                        fd.as_vector([0, -1, 0]),
                        fd.as_vector([0, 0, 0]),),
                    fd.as_vector([0, 0, 0]),),
                fd.as_vector([0, 0, 0]),)
            '''
            fd.conditional(fd.gt(x, 0.2 - 0.005),
                    fd.conditional(fd.gt(y, 0.05 - 0.005),
                    fd.conditional(
                        fd.lt(y, 0.05 + 0.005),
                        fd.as_vector([0, -1, 0]),
                        fd.as_vector([0, 0, 0]),),
                    fd.as_vector([0, 0, 0]),),
                fd.as_vector([0, 0, 0]),)'''

            # elasticity parameters
            # SIMP based E
            self.E = self.E0 + (self.E1 - self.E0) * (self.rho_hat**self.penalisationExponent)
            lambda_ = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
            mu = (self.E) / (2 * (1 + self.nu))

            # linear elastic weak variational form
            epsilon = 0.5 * (fd.grad(v) + fd.grad(v).T)
            sigma = (lambda_ * fd.div(u) * fd.Identity(self.mesh.geometric_dimension())) + (2 * mu * 0.5 * (fd.grad(u) + fd.grad(u).T))
            a = fd.inner(sigma, epsilon) * fd.dx
            L = fd.dot(T, v) * fd.ds(2)

            # solve
            u = fd.Function(self.vectorFunctionSpace)
            fd.solve(a == L, u, bcs=self.bcs)

            # output displacement visualisation
            self.uFunction.assign(u)
            self.uFile.write(self.uFunction)

            # assemble objective function
            self.j = fd.assemble(fd.inner(T, u) * fd.ds(2))

            # volume fraction constraint
            self.c1 = 0.3 - ((1 / self.meshVolume) * fd.assemble(self.rho_hat * fd.dx))

            # compute objective function sensitivities
            self.djdrho = (
                fda.compute_gradient(self.j, fda.Control(self.rho)).vector().get_local()
            ) / self.gradientScale

            # compute constraint sensitivities
            self.dc1drho = (
                fda.compute_gradient(self.c1, fda.Control(self.rho)).vector().get_local()
            ) / self.gradientScale

            # assemble constraint vector
            self.c = np.array([self.c1])

            # assemble jacobian vector
            # self.dcdrho = np.concatenate((self.dc1drho, self.dc2drho))
            self.dcdrho = self.dc1drho

        else:
            pass

        return self.j, self.djdrho, self.c, self.dcdrho