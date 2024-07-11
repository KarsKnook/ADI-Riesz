import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from mpmath import ellipfun
from tabulate import tabulate

def MMS(U):
    N = len(U)
    F = np.zeros((N,N,N))
    kaas = np.zeros((N**2, N))

    for i in range(N):
        kaas[N*i:N*(i+1),:] += sum(mass[i,j]*mass@U[:,:,j]@mass for j in range(N))
        kaas[N*i:N*(i+1),:] += sum(mass[i,j]*weak_laplacian@U[:,:,j]@mass for j in range(N))
        kaas[N*i:N*(i+1),:] += sum(mass[i,j]*mass@U[:,:,j]@weak_laplacian for j in range(N))
        kaas[N*i:N*(i+1),:] += sum(weak_laplacian[i,j]*mass@U[:,:,j]@mass for j in range(N))

    kaas = np.linalg.solve(np.kron(mass, np.eye(N)), kaas)

    for i in range(N):
        F[:,:,i] = np.linalg.solve(mass, kaas[N*i:N*(i+1),:])
        F[:,:,i] = np.linalg.solve(mass.T, F[:,:,i].T).T

    return F

def mobius(z, a, b, c, d, alpha):
    t1 = a*(-alpha*b + b + alpha*c + c) - 2*b*c
    t2 = a*(alpha*(b+c) - b + c) - 2*alpha*b*c
    t3 = 2*a - (alpha+1)*b + (alpha-1)*c
    t4 = -alpha*(-2*a+b+c) - b + c

    return (t1*z + t2)/(t3*z + t4)

def ADI_shifts(a, b, c, d, tol):
    gamma = (c-a)*(d-b)/((c-b)*(d-a))
    J = int(np.ceil(np.real(np.log(16*gamma)*np.log(4/tol)/np.pi**2)))
    alpha = -1 + 2*gamma + 2*np.sqrt(gamma**2 - gamma)
    alpha = np.real(alpha)

    K = sp.special.ellipk(1-1/alpha**2)
    dn_ = ellipfun("dn")
    dn = [dn_((2*j + 1)*K/(2*J), 1-1/alpha**2) for j in range(J)]

    return [mobius(-alpha*i, a, b, c, d, alpha) for i in dn], [mobius(alpha*i, a, b, c, d, alpha) for i in dn]

def ADI_generalised(A, B, C, D, F, a, b, c, d, tol):
    "ADI method for solving generalised sylvester AXB - CXD = F"
    n = len(A)
    X = np.zeros((n, n))

    gamma = (c-a)*(d-b)/((c-b)*(d-a))
    J = int(np.ceil(np.real(np.log(16*gamma)*np.log(4/tol)/np.pi**2)))
    p, q = ADI_shifts(a, b, c, d, tol)
    p, q = [float(i) for i in p], [float(i) for i in q]

    for j in range(J):
        X = np.linalg.solve(C, F - (A - p[j]*C)@X@B)
        X = np.linalg.solve((D - p[j]*B).T, X.T).T
        X = np.linalg.solve(A - q[j]*C, F - C@X@(D - q[j]*B))
        X = np.linalg.solve(B.T, X.T).T
    return X


nmax = 2
pmax = 10

table = True
latex = True

rates = np.zeros((nmax, pmax))
n_list = list(range(1,nmax+1))
p_list = list(range(1,pmax+1))

for i,n in enumerate(n_list):
    for j,p in enumerate(p_list):
        print(f"{n} x {n} x {n} cells up to degree {p}")
        mass = np.load(f"matrices/mass_{n}_{p}.npz")
        weak_laplacian = np.load(f"matrices/weak_laplacian_{n}_{p}.npz")

        N = len(mass)

        U = np.random.rand(N,N,N)
        F = MMS(U)

        c = 0.5
        d = 0.5 + weak_laplacian[-1, -1]/np.real(sp.sparse.linalg.eigs(mass, k=1, which="SM", return_eigenvectors=False)[0])
        a = -d
        b = -c

        mass_inv = np.linalg.inv(mass)
        mass_inv_kron_I = np.kron(mass_inv, np.eye(N))
        fv1 = np.zeros((N**2, N))
        fv2 = np.zeros((N**2, N))
        fv3 = np.zeros((N**2, N))
        for k in range(N):
            fv1[N*k:N*(k+1),:] = sum(mass[k,l]*mass@F[:,:,l]@mass for l in range(N))
            fv2[N*k:N*(k+1),:] = sum(mass[k,l]*mass@F[:,l,:]@mass for l in range(N))
            fv3[N*k:N*(k+1),:] = sum(mass[k,l]*mass@F[l,:,:]@mass for l in range(N))

        U_num = np.zeros((N,N,N))
        error_list = [np.linalg.norm(U - U_num)]

        for k in range(10):
            # dz on RHS
            RHS = np.copy(fv1)
            for l in range(N):
                RHS[N*l:N*(l+1),:] -= sum(weak_laplacian[l,m]*mass@U_num[:,:,m]@mass for m in range(N))
            Y = mass_inv_kron_I@RHS
            for l in range(N):
                U_num[:,:,l] = ADI_generalised(0.5*mass + weak_laplacian, mass, -mass, 0.5*mass + weak_laplacian, Y[N*l:N*(l+1),:], a, b, c, d, 1e-20)

            # dy on RHS
            RHS = np.copy(fv2)
            for l in range(N):
                RHS[N*l:N*(l+1),:] -= sum(weak_laplacian[l,m]*mass@U_num[:,m,:]@mass for m in range(N))
            Y = mass_inv_kron_I@RHS
            for l in range(N):
                U_num[:,l,:] = ADI_generalised(0.5*mass + weak_laplacian, mass, -mass, 0.5*mass + weak_laplacian, Y[N*l:N*(l+1),:], a, b, c, d, 1e-20)

            # dx on RHS
            RHS = np.copy(fv3)
            for l in range(N):
                RHS[N*l:N*(l+1),:] -= sum(weak_laplacian[l,m]*mass@U_num[m,:,:]@mass for m in range(N))
            Y = mass_inv_kron_I@RHS
            for l in range(N):
                U_num[l,:,:] = ADI_generalised(0.5*mass + weak_laplacian, mass, -mass, 0.5*mass + weak_laplacian, Y[N*l:N*(l+1),:], a, b, c, d, 1e-20)

            error_list.append(np.linalg.norm(U - U_num))

        #plt.plot(range(11), error_list, label=f"n={n}, p={p}, rate={np.round((error_list[-1]/error_list[0])**(1/10), 3)}")
        rates[i,j] = np.round((error_list[-1]/error_list[0])**(1/10), 3)

# ------------------------------- Print iteration table -------------------------------------
if table:
    table_headers = ["n \\ p", *[str(p) for p in p_list]]
    table_data = []

    for i,n in enumerate(n_list):
        row = [f"{n}"]
        for j,p in enumerate(p_list):
            row.append(f"{rates[i,j]}")

        table_data.append(row)

    print(tabulate(table_data, headers=table_headers, tablefmt="fancy_grid"), flush=True)


# --------------------------- Print to LaTeX table format --------------------------------
if latex:
    out = "n \\textbackslash p" + "".join([f"& {p} " for p in p_list]) + " \\\\ \n"
    out += "\hline \n"

    for i,n in enumerate(n_list):
        row = f"{n}"
        for j,p in enumerate(p_list):
            row += f"& {rates[i,j]}"
        row += " \\\\ \n"
        out += row

    print(out, flush=True)
