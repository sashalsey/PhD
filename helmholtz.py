import firedrake as fd # type: ignore
nelx, nely, nelz = 20, 10, 10
lx, ly, lz = 1, 1, 1
Vol = lx*ly*lz
mesh = fd.BoxMesh(nelx, nely, nelz, lx, ly, lz, hexahedral=True)
R = fd.FunctionSpace(mesh, "DG", 1)  # Density space

def helmholtz_filter(X,rlen):
    Xf = fd.TrialFunction(R)
    w = fd.TestFunction(R)
    a = (fd.inner((rlen**2)*fd.grad(Xf),fd.grad(w))*fd.dx +fd.inner(Xf,w)*fd.dx)
    L = fd.inner(X,w)*fd.dx
    Xf = fd.Function(R)
    fd.solve(a == L, Xf)
    return Xf
