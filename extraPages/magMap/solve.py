import torch, torch.nn.functional as F, time

'''
Mostly just some linear algebra and calculus review.
Also some from https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
'''


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

        # if debug: print(' - residual norm at ',i, (b-A@x).norm())

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
        print(f' - first 8 ds A-conjugacy:\n{ds@(A@ds.mT)}') # The point is to have zeros on the off-diagonals here
    return x

def solve_cg_AtA(A, b, x0, T, preconditioned=False, debug=False):
    x = x0
    Z = A.mT @ b
    W = A.mT @ (A @ x)
    r = A.mT @ b - A.mT @ (A @ x)

    # Minv = 1./get_diag(A.mT@A) if preconditioned else None
    # Minv = 1./get_diag(A) if preconditioned else None
    # FIXME: Test this: instead of forming big/less-sparse A'A, approximate diag(A'A) ~=~ A_ii^2.
    #        But is this too inaccurate in some cases?
    Minv = 1./(get_diag(A)**2) if preconditioned else None

    d = r if not preconditioned else Minv * r

    T = min(T, len(b)*2)

    if debug: print(' - residual norm at start', (A.mT@b-A.mT@(A@x)).norm())
    st = time.time()

    # FIXME: Shewchuk avoids r1 by storing the inner product for the next iter (he calls delta_new and delta_old)
    for i in range(T):
        # WARNING: Interesting the order of the mult changes accuracy of CG/PCG (and more difference when sparse)
        # a = ((r@r) if not preconditioned else (r@(Minv*r))) / (d@A.mT@(A@d))
        a = ((r@r) if not preconditioned else (r@(Minv*r))) / (d@(A.mT@(A@d)))

        x = x + a * d
        r1 = r - a * (A.mT@(A@d))
        if not preconditioned:
            beta = (r1 @ r1) / (r@r)
            d = r1 + beta * d
        else:
            s = Minv*r1
            beta = (r1 @ s) / (r@(Minv*r))
            d = s + beta * d

        r = r1
        if r.norm() < 1e-7: break

    if debug: print(' - residual norm at   end', (A.mT@b-A.mT@(A@x)).norm())
    if debug: print(' - {} steps in {:.2f}ms'.format(i, (time.time() - st)*1000))

    return x
