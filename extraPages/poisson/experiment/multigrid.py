import torch, torch.nn.functional as F, numpy as np
from scipy.sparse import coo_array
from scipy.sparse.linalg import spsolve

def torchCoo_to_scipy(st):
    return coo_array(( st.values().cpu().numpy(), (st.indices()[0].cpu().numpy(), st.indices()[1].cpu().numpy())), shape = st.shape[:2]).tocsr()

def show(solvers):
    pass

class Solver:
    def __init__(self, name):
        self.name = name

    def run(self, graphSt, x0):
        raise NotImplementedError()

class SparseExactSolver(Solver):
    def __init__(self):
        super().__init__('sparse_exact')


    def run(self, graphSt, bias, x0):
        A = torchCoo_to_scipy(graphSt)
        # print(A.toarray())
        b = bias.cpu().numpy()
        x1 = spsolve(A,b)
        return torch.from_numpy(x1)


def main():
    cyclic = False
    # N = 100
    N = 6
    sz = (N,N)
    grid = torch.arange(N).view(1,-1)

    unaryStrength = 100
    binStrength = 1
    bias = torch.zeros(N)
    bias[0]   = 1 * (unaryStrength * unaryStrength)
    bias[N-1] = 2 * (unaryStrength * unaryStrength)

    if 0:
        csts = []

        # binary constraints
        for i in range(N-1):
            bin = torch.zeros(N)
            bin[i  ] = 1
            bin[i+1] = -1
            csts.append(binStrength*bin)

        # unary constraints
        csts.append(unaryStrength*F.one_hot(torch.LongTensor((0,)), N)[0])
        csts.append(unaryStrength*F.one_hot(torch.LongTensor((N-1,)), N)[0])

        csts = torch.stack(csts,0).to_sparse()
        print(csts)
        d=csts
    else:

        # This is VERY unintuitive. The sparse tensor is describing the stacked jacobian/constraint blocks.
        # So we need to increment the current row and append as many entries as constraint-columns.
        # Very confusing...

        i = torch.LongTensor((0,0, 1,N-1)).view(-1,2)
        v = torch.ones(2) * unaryStrength
        numCsts = 2 # Already have two unary constriants

        # binary constraints (2 non-zero columns, but increment row once)
        for j in range(N-1):
            i = torch.cat((i,torch.LongTensor((numCsts,j  )).view(1,2)),0)
            i = torch.cat((i,torch.LongTensor((numCsts,j+1)).view(1,2)),0)
            v = torch.cat((v,torch.FloatTensor((1,-1))*binStrength), 0)
            numCsts += 1

        d = torch.sparse_coo_tensor(i.view(-1,2).T,v)

    lapPlusAnchors = d.mT @ d

    '''
    lap = d @ d.mT
    # print('lap\n',lap.to_dense())


    anchorsSt = torch.sparse_coo_tensor(
        torch.LongTensor((0,0, N-1,N-1)).view(-1,2).T,
        torch.ones(2),
        size=sz)

    lapPlusAnchors = (lap + anchorsSt).coalesce()
    '''

    x0 = torch.zeros(N)
    # print('graph\n',lapPlusAnchors.to_dense())

    y = SparseExactSolver().run(lapPlusAnchors, bias, x0)
    print(' - exact soln   :', y)
    print(' - residual     :', (lapPlusAnchors @ y - bias))
    print(' - residual norm:', (lapPlusAnchors @ y - bias).norm())


main()
