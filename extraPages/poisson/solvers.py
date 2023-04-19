import torch, torch.nn.functional as F, time

'''
Mostly just some linear algebra and calculus review.
Also some from https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
'''


# Generates a rotation matrix (det=+1 or -1, but evalues may be imaginary and/or negative)
# NOTE: Input is expected to be a row-major set of vectors.
#       Output is an orthogonal set of column vectors (also row ortho...).
def gram_schmidt(vs):
    N = len(vs)
    B = torch.zeros((N,N))
    for i in range(N):
        u = vs[i]
        for j in range(i):
            u -= B[:,j] * (B[:,j] @ u) / (B[:,j]@B[:,j])
        B[:,i] = u
    return B / B.norm(dim=0,keepdim=True)

# Section 7.2.
# Like above, but vectors are A-conjugate, not orthogonal w.r.t standard basis.
def gram_schmidt_conjugate(vs, A):
    N = len(vs)
    B = torch.zeros((N,N))
    for i in range(N):
        u = vs[i]
        for j in range(i):
            u -= B[:,j] * (u @A@ B[:,j]) / (B[:,j]@A@B[:,j])
        B[:,i] = u
    return B / B.norm(dim=0,keepdim=True)

# Test gram schmidt, work on generating random symmetric-PD matrix
if 0:
    A = torch.FloatTensor((
        1,0,0,
        0,2,2,
        1,2,1)).reshape(3,3)
    A = torch.randn(4,4)
    # A = (A@A.mT)
    # A = (A+A.mT)
    print(f' - A:\n{A}')
    B = gram_schmidt(A)
    evalues = torch.linspace(.1,5,4) ** 2
    B = B @ evalues.diag() @ B.mT
    print(f' - B:\n{B}')
    print(f' - B@B.T:\n{B@B.mT}')
    print(f' - B.T@B:\n{B.mT@B}')
    print(f' - Eig{{B}}:\n{torch.linalg.eig(B)}')
    print(f' - Det{{B}}: {torch.linalg.det(B)}')
    max_div_min = lambda t: (t.max() / t.min()).item()
    print(f' - Condition{{B}} {max_div_min(torch.linalg.eig(B).eigenvalues.real)}')
    exit()

# Look at conjugate GS
if 0:
    A = torch.FloatTensor((
        1,0,1,
        0,5,0,
        1,0,4)).reshape(3,3) # In a least-squares setting, the matrix will be symmetric-PD
    # A = torch.eye(3) # This is really the simplest check.

    V = torch.FloatTensor((
        1,0,0,
        0,2,2,
        1,2,1)).reshape(3,3)

    B = gram_schmidt_conjugate(V,A)
    # evalues = torch.linspace(.1,5,3) ** 2
    # B = B @ evalues.diag() @ B.mT
    print(f' - B:\n{B}')
    print(f' - B@B.T:\n{B@B.mT}')
    print(f' - B.T@B:\n{B.mT@B}')
    print(f' - Eig{{B}}:\n{torch.linalg.eig(B)}')
    print(f' - Det{{B}}: {torch.linalg.det(B)}')
    max_div_min = lambda t: (t.max() / t.min()).item()
    print(f' - Condition{{B}} {max_div_min(torch.linalg.eig(B).eigenvalues.real)}')
    # WARNING: Orthonormality applies to both rows and columns.
    # WARNING: But conjugacy does not. The GS process's outputs are only necessarily column-wise conjugate!
    print(f' - A-conjugacy:\n{B.mT@A@B}')
    exit()


def get_problem_1():

    if 0:
        A = torch.FloatTensor((150,21,21,3.)).view(2,2)
        b = torch.FloatTensor((2,-8.))
    else:
        # A random but spectrum-controlled symmetric-PD matrix.
        torch.manual_seed(0)
        N = 4
        A = torch.randn(N,N)
        A = (A@A.mT)
        A = gram_schmidt(A)
        evalues = torch.linspace(.1, 5, N) ** 2
        A = A @ evalues.diag() @ A.mT
        b = torch.randn(N)
    x = torch.linalg.solve(A,b)

    print('')
    print(f' - A\n{A}')
    print(f' - Eig{{A}} {torch.linalg.eig(A).eigenvalues.real}')
    max_div_min = lambda t: (t.max() / t.min()).item()
    print(f' - Condition{{A}} {max_div_min(torch.linalg.eig(A).eigenvalues.real)}')
    print(f' - Ground-Truth Solved x {x}')
    print('')

    return A, b, x


def run_generic(A, b, x, T, x0, solve_func, **kw):
    def getError(xi): return (x-xi).norm()
    def getRes(xi): return b-A@xi

    print(f'error0 {getError(x0)}')
    print(f'res0   {getRes(x0)}')

    xi = solve_func(A, b, x0, T, **kw)

    print(f'errorT {getError(xi)}')
    print(f'resT   {getRes(xi)}')

def solve_steepest_descent(A, b, x0, T):
    xi = x0
    for i in range(T):
        r = b - A@xi
        # a = (r@r) / (r@A@r + 1e-9)
        a = (r@r) / ((A@r)@r.T + 1e-9)
        xi1 = xi + a*r
        xi = xi1
        if r.norm() < 1e-7: break
    return xi

def solve_jacobi(A, b, x0, T):
    if A.is_sparse:
        print(' - Jacobi not implemented for sparse matrices')
        return x0

    # Split Matrix
    D = A * torch.eye(A.size(0))
    E = A - D

    # Form iteration matrices
    Dinv = (1./torch.diag(D)).diag()
    B = -Dinv @ E
    z = Dinv @ b

    print(f'Spectral Radius of \'B\' (max evalue, smaller better) ---> {torch.linalg.eig(B).eigenvalues.abs().max()}')

    xi = x0
    for i in range(T):
        xi = B @ xi + z
    return xi

def get_diag(A):
    if A.is_sparse:
        n = A.size(0)
        S = torch.sparse_coo_tensor(torch.arange(n).unsqueeze_(0).repeat(2,1).to(A.device), torch.ones(n,device=A.device), size=(n,n))
        B = (A * S).coalesce()
        d = torch.zeros(n, device=A.device, dtype=A.dtype)
        d[B.indices()[0]] = B.values() # ind[0] = ind[1] so this is correct
        return d
    else:
        return A.diag()

# Only allows jacobi (diagonal) preconditioning
def solve_cg(A, b, x0, T, preconditioned=False, debug=False, debugConj=False):
    x = x0
    r = b - A @ x
    Minv = 1./get_diag(A) if preconditioned else None
    # Minv = torch.eye(x.size(0)).diag()
    d = r if not preconditioned else Minv * r

    ds = []
    rs = []

    T = min(T, len(b)*2)

    if debug: print(' - residual norm at start', (b-A@x).norm())
    st = time.time()

    # FIXME: Shewchuk avoids r1 by storing the inner product for the next iter (he calls delta_new and delta_old)
    for i in range(T):
        if debugConj: ds.append(d); rs.append(r)

        # WARNING: Interesting the order of the mult changes accuracy of CG/PCG (and more difference when sparse)
        # a = ((r@r) if not preconditioned else (r@(Minv*r))) / (d@A@d)
        a = ((r@r) if not preconditioned else (r@(Minv*r))) / (d@(A@d))

        x = x + a * d
        r1 = r - a * (A@d)
        if not preconditioned:
            beta = (r1 @ r1) / (r@r)
            d = r1 + beta * d
        else:
            s = Minv*r1
            beta = (r1 @ s) / (r@(Minv*r))
            d = s + beta * d

        r = r1
        if r.norm() < 1e-7: break

    if debug: print(' - residual norm at   end', (b-A@x).norm())
    if debug: print(' - {} steps in {:.2f}ms'.format(i, (time.time() - st)*1000))

    if debugConj:
        ds = torch.stack(ds[:8],0)
        rs = torch.stack(rs[:8],0)
        # print(f' - first 8 ds A-conjugacy:\n{ds@A@ds.mT}') # The point is to have zeros on the off-diagonals here
        # print(f' - first 8 ds A-conjugacy:\n{(A@ds.mT)@ds}') # The point is to have zeros on the off-diagonals here
        print(f' - first 8 ds A-conjugacy:\n{ds@(A@ds.mT)}') # The point is to have zeros on the off-diagonals here
        # print(f' - rs A-conjugacy:\n{rs@A@rs.mT}')
    return x




if __name__ == '__main__':
    torch.set_printoptions(linewidth=120, sci_mode=False)
    A, b, x = get_problem_1()

    # Sparsify
    A = A.to_sparse()

    N = b.size(0)
    x0 = torch.full((N,), -2.)
    T = 100

    print('\n**********************************')
    print(' Running Steepest Descent')
    run_generic(A, b, x, T, x0, solve_func=solve_steepest_descent)
    print('\n**********************************')
    print(' Running Jacobi')
    run_generic(A, b, x, T, x0, solve_func=solve_jacobi)
    print('\n**********************************')
    print(' Running CG')
    run_generic(A, b, x, T, x0, solve_func=solve_cg)
    print('\n**********************************')
    print(' Running PCG')
    run_generic(A, b, x, T, x0, solve_func=solve_cg, preconditioned=True, debug=True, debugConj=True)
