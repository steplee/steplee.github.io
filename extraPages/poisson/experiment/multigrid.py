import torch, torch.nn.functional as F, numpy as np
from scipy.sparse import coo_array
from scipy.sparse.linalg import spsolve, cg

import matplotlib.pyplot as plt

# This MG solver works, but since it is attempting to accelerate an iterative method, it needs
# a good iterative method in the first place.
# None of the iterative algos are that good, even CG with 64 iters is not great.
#
# The interpolation problem here is "stiff": the set of eigenvalues has high dynamic range.
# Maybe PCG would be good.
#
# Note: solvers should take matrix in constructor, so they can preproeces..

def torchCoo_to_scipy(st):
    return coo_array(( st.values().cpu().numpy(), (st.indices()[0].cpu().numpy(), st.indices()[1].cpu().numpy())), shape = st.shape[:2]).tocsr()

def smoothstep(lo,hi,x):
    t = np.clip((x-lo)/(hi-lo), 0,1)
    return t * t * (3 - 2 * t)

def normalize_rows(a):
    n = a.size(0)
    diag = torch.sparse_coo_tensor(torch.arange(n).repeat(2).view(2,n), torch.ones(n),size=(n,n)).coalesce()
    aa = (a@a.mT).coalesce()
    aa = torch.sparse_coo_tensor(aa.indices(),aa.values(),size=aa.size()).coalesce()
    diag = diag * aa
    diag = torch.sparse_coo_tensor(diag.indices(), 1./diag.values().sqrt(), size=diag.size())
    return diag @ a

def get_diag(a):
    n = a.size(0)
    diag = torch.sparse_coo_tensor(torch.arange(n).repeat(2).view(2,n), torch.ones(n),size=(n,n)).coalesce()
    a = torch.sparse_coo_tensor(a.indices(),a.values(),size=a.size()).coalesce()
    return (diag * a).coalesce()



def show1(x,y, unary, binary, res, title, figAx=None):

    if figAx is None:
        fig,ax = plt.subplots(1)
        fig.suptitle(title)
        fig.canvas.manager.set_window_title(title)
    else:
        fig,ax = figAx

    ax.plot(x,y)

    if unary is not None:
        unary_x = np.array([x[j] for j,tgt,str in unary])
        unary_y = np.array([tgt  for j,tgt,str in unary])
        unary_s = np.array([str  for j,tgt,str in unary])
        ax.scatter(unary_x, unary_y, s=29.5, marker='o', edgecolors='orange',facecolors='none')

    if binary is not None:
        binary_x1 = np.array([x[j] for j,k,tgt,str in binary])
        binary_x2 = np.array([x[k] for j,k,tgt,str in binary])
        binary_y1 = np.array([y[j] for j,k,tgt,str in binary])
        binary_t  = np.array([tgt  for j,k,tgt,str in binary])
        binary_s  = np.array([str  for j,k,tgt,str in binary])
        for x1,x2,y1,tgt,s in zip(binary_x1,binary_x2,binary_y1,binary_t,binary_s):
            ax.arrow(x1,y1, x2-x1, -tgt, color='magenta', width=smoothstep(.5,50,s)*.009, zorder=5)

    # Show residuals
    if res is not None:
        if isinstance(res,torch.Tensor): res = res.cpu().numpy()
        resn = np.linalg.norm(res)
        # res = -1./np.log(1e-9+abs(res))
        # res = res / np.max(abs(res))
        # res = res * abs(y).max().item()
        ax.plot(x,res, alpha=.5, color=(.9,.5,.4))

def showN(unary,binary, xs,ys,ress, title, figAx):
    L = len(xs)
    for l in range(L):
        show1(xs[l], ys[l], unary[l],binary[l], ress[l], title, figAx=(figAx[0],figAx[1][l]))

class Solver:
    def __init__(self, name):
        self.name = name

    def run(self, graphSt, x0):
        raise NotImplementedError()

class SparseExactSolver(Solver):
    def __init__(self):
        super().__init__('SparseExact')

    def run(self, graphSt, bias, x0):
        A = torchCoo_to_scipy(graphSt)
        # print(A.toarray())
        b = bias.cpu().numpy()
        x1 = spsolve(A,b)
        return torch.from_numpy(x1)

# This actually gets stuck in a mode where the residuals-vector flips for its negative...
class SparseCgSolver(Solver):
    def __init__(self):
        super().__init__('SparseExact')
        self.M = None

    def run(self, graphSt, bias, x0):
        A = torchCoo_to_scipy(graphSt)
        b = bias.cpu().numpy()
        # x1,code = cg(A,b).numpy())

        if self.M is None:
            self.M = np.diag(1./get_diag(graphSt).values().cpu().numpy())
        x1,code = cg(A,b, x0=x0.cpu().numpy(), maxiter=24, M=self.M)

        # x1,code = cg(A,b, x0=x0.cpu().numpy(), maxiter=64)
        # x1,code = cg(A,b, x0=x0.cpu().numpy(), maxiter=64*4)
        # x1,code = cg(A,b, maxiter=32)
        # print(code)
        return torch.from_numpy(x1)

# This is quite unstable unless alpha<=.7
class JacobiSolver(Solver):
    def __init__(self, iters=1000, alpha=.7):
        super().__init__(f'Jacobi_i={iters}_a={alpha:.2f}')
        self.iters = iters
        self.alpha = alpha
        self.O = None

    def run(self, A, b, x0):
        M,N = A.size(0), A.size(1)
        x = x0.clone()

        # D = A * torch.sparse_coo_tensor(torch.arange(N).view(1,N).repeat(2,1), torch.ones(N), size=A.size())
        if self.O is None:
            offdiag_mask = A.indices()[0] != A.indices()[1]
            O = torch.sparse_coo_tensor(
                A.indices()[:,offdiag_mask],
                A.values()[offdiag_mask],
                size=A.size()).coalesce()
            Dinv = torch.sparse_coo_tensor(
                A.indices()[:,~offdiag_mask],
                1./A.values()[~offdiag_mask],
                size=A.size()).coalesce()
            self.O,self.Dinv = O,Dinv
        else:
            O,Dinv =self.O,self.Dinv


        for i in range(self.iters):
            # x = x + (Dinv @ (b - O @ x))
            # x = x*.3 + .7*(Dinv @ (b - O @ x))
            x = x*(1-self.alpha) + self.alpha*(Dinv @ (b - O @ x))

        return x

class SteepestDescentSolver(Solver):
    def __init__(self, iters=1000, alpha=.9):
        super().__init__(f'Sd_i={iters}_a={alpha:.2f}')
        self.iters = iters
        self.alpha = alpha

    def run(self, A, b, x0):
        M,N = A.size(0), A.size(1)
        x = x0.clone()


        for i in range(self.iters):
            r = b - A@x
            a = (r@r) / (r@(A@r))
            x = x + self.alpha * a * r

        return x

class MultigridSolverV(Solver):
    def __init__(self):
        super().__init__('MG-V')
        self.lvls = 5
        self.doShow = True
        self.doShow = False
        self.doShowEnd = True

        self.figAxs = plt.subplots(self.lvls)
        if self.lvls == 1: self.figAxs = (self.figAxs[0], [self.figAxs[1]])

        self.overRelax = 1.1

    def run(self, A, b, x0, **vizKw):
        self.vizKw = vizKw
        M,N = A.size(0), A.size(1)
        x = x0.clone()

        self.K = torch.FloatTensor((.1,.4,.4,.1)).view(1,1,-1)
        self.K = torch.FloatTensor((.1,.1,.1,.1)).view(1,1,-1)
        self.K.div_(self.K.sum())

        self.levels = []
        self.rs = []
        self.xs = []
        self.Xs = []
        self.to_next = []
        self.to_prev = []
        Ai,bi = A, b
        for i in range(self.lvls):

            Ni = Ai.size(0)

            # solveri = JacobiSolver(iters=20,alpha=.7)
            # solveri = SteepestDescentSolver(iters=20,alpha=.7)

            # Intersetingly, jaobi works well as solver on high-res, but SD works better on lower res
            solveri = JacobiSolver(iters=40,alpha=.7)
            # solveri = SparseCgSolver()
            # if i > 0: solveri = SteepestDescentSolver(iters=50,alpha=.7)
            if i > 0: solveri = SparseCgSolver()

            # solveri = SparseCgSolver()

            self.levels.append((Ai,bi,solveri))
            self.xs.append(torch.zeros(Ni))
            self.rs.append(torch.zeros(Ni))
            self.Xs.append(np.linspace(0,1,Ni))

            toi,tov = [],[]
            for k in range(Ni//2):
                v = []
                if k > 0:
                    toi.extend([k,k*2-2])
                    v.append(self.K[0,0,0])
                if k > 0:
                    toi.extend([k,k*2-1])
                    v.append(self.K[0,0,1])
                if k < Ni//2-1:
                    toi.extend([k,k*2])
                    v.append(self.K[0,0,2])
                if k < Ni//2-1:
                    toi.extend([k,k*2+1])
                    v.append(self.K[0,0,3])
                tov.extend((np.array(v)/sum(v)).tolist())
            toi = torch.LongTensor(toi).view(-1,2).T
            # print(toi)
            # toi = torch.LongTensor(toi).view(2,-1)
            # exit()
            # tov = torch.FloatTensor(tov) * np.sqrt(2)
            tov = torch.FloatTensor(tov) * 1
            # to_next = torch.sparse_coo_tensor(toi,tov,size=(Ni//2,Ni)).coalesce()
            to_next = normalize_rows(torch.sparse_coo_tensor(toi,tov,size=(Ni//2,Ni)).coalesce())
            to_prev = normalize_rows(torch.sparse_coo_tensor(toi[[1,0]],tov,size=(Ni,Ni//2)).coalesce())
            # to_next = torch.sparse_coo_tensor(toi,tov,size=(Ni,Ni//2)).coalesce().mT

            # to_next = torch.eye(Ni).view(Ni//2,2,Ni).sum(1).to_sparse()
            to_next = torch.eye(Ni).view(Ni//2,2,Ni).mean(1).to_sparse() # NOTE:

            self.to_next.append(to_next)
            self.to_prev.append(to_prev)
            print('to_next:\n',to_next)

            if i < self.lvls-1:
                Ai = to_next @ Ai @ to_next.mT
                bi = to_next @ bi # Not actually important.


        # V-cycles
        for i in range(150):
            x = self.go(x,b,0, **vizKw, lastIter=i==149)

        return x

    def smooth(self, x, f, lvl):
        A_,b_,solver = self.levels[lvl]
        # return solver.run(A_,b_,x)
        return solver.run(A_,f,x)

    def residual(self, x, f, lvl):
        A_,b_,solver = self.levels[lvl]
        # return A_@x-b_
        # return (A_@x-f)
        return (f-A_@x)

    def restrict(self, f, lvl):
        # return f.view(-1,2).sum(1)
        return f.view(-1,2).mean(1)
        return self.to_next[lvl] @ f * 1
        # return self.to_next[lvl] @ f * .5
        # return F.conv1d(f.view(1,1,-1), self.K, padding=2,stride=2)[0,0]#.view(-1,2)[:,0]
        # return f.view(-1,2).mean(1)

    def prolong(self, eps, lvl):
        # return F.conv_transpose1d(eps.view(1,1,-1), self.K, padding=1,stride=2)[0,0]#.repeat_interleave(2)
        # return eps.repeat_interleave(2) * 2
        # return eps.repeat_interleave(2) * .5
        # return eps.repeat_interleave(2) * 2
        # return (self.to_next[lvl].mT @ eps) * .1
        # return (self.to_next[lvl].mT @ eps) * 1
        # return (self.to_next[lvl].mT @ eps) * 1
        # return (self.to_next[lvl].mT @ eps) * 2
        # return (self.to_next[lvl].mT @ eps) * .5
        return (self.to_prev[lvl] @ eps) * 1 # WHY does over relaxationg help?
        # return F.conv_transpose1d(eps.view(1,1,-1), self.K, padding=2)[0,0].repeat_interleave(2) *.5
        # return F.conv_transpose1d(eps.view(1,1,-1), self.K, padding=1, stride=2)[0,0,1:]


    def go(self, x, f0, lvl, **vizKw):
        f=f0.clone()

        # Pre-smooth
        x = self.smooth(x,f,lvl)

        # Compute residuals
        r = self.residual(x,f,lvl)
        # r=self.levels[lvl][1]
        print(f' - MG-V down lvl={lvl}, r={r.norm().item():.5f}')

        # Restrict
        rhs = self.restrict(r, lvl)

        eps = torch.zeros_like(rhs)

        # Recurse or not
        if lvl >= self.lvls-2:
            if self.lvls > 1:
                eps = self.smooth(eps,rhs,1+lvl)
                self.xs[lvl+1] = eps
                self.rs[lvl+1] = self.residual(eps,rhs,lvl+1)
        else:
            eps = self.go(eps,rhs,1+lvl)

        # Prolong
        # if lvl == 0:
        # print('PROLONG', eps.shape,self.prolong(eps,lvl).shape)
        # x = x - self.prolong(eps, lvl)
        # print('BEFORE\n',x)
        # x = x + self.prolong(eps, lvl) * self.overRelax
        x = x + self.prolong(eps, lvl)
        # print('AFTER\n',x)
        # x = self.prolong(eps, lvl)

        # Final smooth
        x = self.smooth(x,f,lvl)
        self.xs[lvl] = x
        self.rs[lvl] = self.residual(x,f,lvl)
        print(f' - MG-V up   lvl={lvl}, r={self.rs[lvl].norm().item():.5f}')


        if self.doShow or (self.doShowEnd and vizKw.get('lastIter',False)):
            for ax in self.figAxs[1]:
                ax.clear()

            # def showN(unary,binary, xs,ys,ress, title):
            ress = [None,]*self.lvls
            for i in range(self.lvls):
                # ress[i]=self.residual(self.xs[i], self.levels[i][1], i)
                ress[i]=self.rs[i]

            unary = [self.vizKw['unary']] + [None]*(self.lvls-1)
            binary = [self.vizKw['binary']] + [None]*(self.lvls-1)

            showN(xs=self.Xs,ys=self.xs,title=self.name,ress=ress, unary=unary,binary=binary, figAx=self.figAxs)
            plt.show()
            plt.waitforbuttonpress()

        return x



def main():
    cyclic = False
    N = 128*4
    # N = 5
    sz = (N,N)
    grid = torch.arange(N).view(1,-1)

    unaryStrength = 100
    # unaryStrength = 1
    binStrength = 1

    # Create the design matrix (aka. stacked jacobian blocks of each square-error constraint)
    # The least squares soln will then be `(J'J)^-1 J' u`.
    # This is rewritten as Ax=b and solved, using A = `J'J` and b = `J'u`
    if 1:
        i = torch.empty(0, dtype=torch.long)
        v = torch.empty(0, dtype=torch.float)
        u = torch.empty(0, dtype=torch.float)
        numCsts = 0

        def add_unary_constraint(j, cstTarget, strength):
            nonlocal i,v,u,numCsts
            i = torch.cat((i,torch.LongTensor((numCsts,j)).view(1,2)),0)
            v = torch.cat((v,torch.FloatTensor((strength,))))
            u = torch.cat((u,strength*torch.FloatTensor((cstTarget,))))
            numCsts += 1
        def add_binary_constraint(j,k, cstTarget, strength):
            nonlocal i,v,u,numCsts
            i = torch.cat((i,torch.LongTensor((numCsts,j)).view(1,2)),0)
            i = torch.cat((i,torch.LongTensor((numCsts,k)).view(1,2)),0)
            v = torch.cat((v,torch.FloatTensor((strength,-strength))))
            u = torch.cat((u,strength*torch.FloatTensor((cstTarget,))))
            numCsts += 1
        def add_second_diff_constraint(j, cstTarget, strength):
            nonlocal i,v,u,numCsts,N
            if j == 0:
                i = torch.cat((i,torch.LongTensor((numCsts,numCsts,j,j+1)).view(2,2).T),0)
                v = torch.cat((v,torch.FloatTensor((strength,-strength))))
                u = torch.cat((u,strength*torch.FloatTensor((cstTarget,))))
            elif j == N-1:
                i = torch.cat((i,torch.LongTensor((numCsts,numCsts,j,j-1)).view(2,2).T),0)
                v = torch.cat((v,torch.FloatTensor((-strength,strength))))
                u = torch.cat((u,strength*torch.FloatTensor((cstTarget,))))
            else:
                i = torch.cat((i,torch.LongTensor((numCsts,numCsts,numCsts,j-1,j,j+1)).view(2,3).T),0)
                v = torch.cat((v,torch.FloatTensor((-strength,2*strength,-strength))))
                u = torch.cat((u,strength*torch.FloatTensor((cstTarget,))))
            numCsts += 1

        # Create unary constraints
        unary = [
            (0, 0, unaryStrength),
            (N//3, 2, unaryStrength),
            (N-1, 1, unaryStrength),
        ]
        # add_unary_constraint(0, 0, unaryStrength)
        # add_unary_constraint(N-1, 1, unaryStrength)
        # add_unary_constraint(N//3, 2, unaryStrength)
        for j, target, strength in unary: add_unary_constraint(j, target, strength)

        # Create binary constraints (That elements j,k differ by this target amount)
        binary = [
            (N//4,N//4+1, -.2, binStrength),
            (N//2,N//2+1, .2, binStrength*10),
            (3*N//4,3*N//4+1, -.1, binStrength),
        ]
        # add_binary_constraint(N//4,N//4+1, -.3, binStrength)
        # add_binary_constraint(N//2,N//2+1, .2, binStrength)
        # add_binary_constraint(3*N//4,3*N//4+1, -.1, binStrength)
        for j,k, target, strength in binary: add_binary_constraint(j,k, target, strength)

        # Create laplacian constraints
        for j in range(N):
            add_second_diff_constraint(j, 0, 1)

        D = torch.sparse_coo_tensor(i.view(-1,2).T,v).coalesce()
        # print(D.to_dense())
        print(D.shape, u.shape)
        # Least-squares-ify
        b = D.mT @ u
        A = D.mT @ D
        print(A.to_dense())

    x0 = torch.zeros(N)
    # print('graph\n',lapPlusAnchors.to_dense())

    X = np.linspace(0,1,N)


    if 1:
        solver = SparseExactSolver()
        y = solver.run(A, b, x0)
        res = A@y-b
        print(f' - {solver.name:24s} residual norm: {res.norm().item():8.6g}')
        show1(X,y, unary,binary,res, title=solver.name)

    if 0:
        solver = JacobiSolver()
        y = solver.run(A, b, x0)
        res = A@y-b
        print(f' - {solver.name:24s} residual norm: {res.norm().item():8.6g}')
        show1(X,y, unary,binary,res, title=solver.name)

    if 0:
        solver = SteepestDescentSolver()
        y = solver.run(A, b, x0)
        res = A@y-b
        print(f' - {solver.name:24s} residual norm: {res.norm().item():8.6g}')
        show1(X,y, unary,binary,res, title=solver.name)

    if 1:
        plt.ion()
        solver = MultigridSolverV()
        y = solver.run(A, b, x0, unary=unary,binary=binary)
        res = A@y-b
        print(f' - {solver.name:24s} residual norm: {res.norm().item():8.6g}')
        show1(X,y, unary,binary,res, title=solver.name)

    plt.show()



main()
