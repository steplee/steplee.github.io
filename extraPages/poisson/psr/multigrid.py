import torch, torch.nn.functional as F
import numpy as np, sys, time
torch.set_printoptions(linewidth=150, edgeitems=10)

from .solvers import solve_cg, solve_cg_AtA
from ..polygonalization.run_marching_algo import run_marching, show_marching_viz

from .singlegrid import get_stencil, get_laplacian, get_convolved_lap, \
        forward_into_grid, backward_from_grid, \
        iter_stencil, iter_stencil_nonzero

from torch.utils.cpp_extension import load
extMod = load('extMod', sources=['poisson/psr/reduce.cu'], extra_cuda_cflags=['--extended-lambda', '-g'], extra_cflags=['-g'])

def test_extMod():
    lvlC = 4
    lvlF = 5
    indsC = torch.randint(0, 1<<lvlC, (3,200)).cuda()
    indsF = torch.randint(0, 1<<lvlF, (3,1000)).cuda()
    # We want them sorted properly...
    indsC = torch.sparse_coo_tensor(indsC,torch.zeros_like(indsC[0])).coalesce().indices()
    indsF = torch.sparse_coo_tensor(indsF,torch.zeros_like(indsF[0])).coalesce().indices()
    stencilSt = get_stencil().to_sparse().cuda()
    A = extMod.make_level_transfer(
            lvlC,lvlF,
            indsC,indsF,
            stencilSt
            )
    print(' - computed A:\n', A)
    exit()
test_extMod()

'''
def get_convolved_lap_halfscale(A):
    d = A.ndim
    L = get_laplacian(A).unsqueeze_(0).unsqueeze_(0)
    A = A.unsqueeze(0).unsqueeze_(0)
    if d == 2: return F.conv2d(L,A,padding=4)[0,0]
    if d == 3: return F.conv3d(L,A,padding=4)[0,0]
    assert False
get_convolved_lap_halfscale(get_laplacian())
'''

# To get Afc (the matrix going from depth c to depth f [always coarse -> fine]),
# we can use this explicit method (which is probably less efficient and clever than others).
# It inflates the lower level by two, then does standard 3d conv.
def get_level_transfer_explicit(stencilSt, stC, stF):
    # output is matrix size [F, C]
    # dBf, dBc = get_laplacian(Bf), get_laplacian(Bc)
    lapC = get_laplacian(cooC)

    Df = int(np.log2(cooF.size(0)))
    Dc = int(np.log2(cooC.size(0)))
    e = Df - Dc
    assert e > 0

    Nf = cooF._nnz()
    Nc = cooC._nnz()

    # The naive way to do this would be to replicate Bc 2^(f-c) times.
    # But since we are implementing sparse conv via iteration over offsets,
    # we can just compute the base and shift each iter and avoid repeating data.

    for sc,sv in iter_stencil_nonzero(stencilSt):
        # Map C -> F_off
        # Map F -> C

        pass



def run_psr(pts0, nrmls0, D=6):
    dev = pts0.device

    # print(f' - stencil {stencil.shape}')
    print(f' - pts {pts.shape}')
    print(f' - nrmls {nrmls.shape}')

    # Map to fixed resolution grid, then de-duplicate entries.
    coo0, off, scale, size = forward_into_grid(pts0, D=D, pad=9)
    st0 = torch.sparse_coo_tensor(coo0.long().t(), nrmls0, size=(size,size,size,3)).coalesce()
    # st0._values(0.div_(st0._values().norm(dim=1,keepdim=True))
    # NOTE: Consider not normalizing, that we we aren't merging bins and also losing samples.
    coo, nrmls2 = st0.indices(), F.normalize(st0.values(), dim=1)

    stencil = get_stencil()
    stencil_st = stencil.to_sparse().coalesce() # A convenient way to iterate over indices/values



if __name__ == '__main__':
    torch.manual_seed(0)
    if 0:
        pts   = torch.randn(512*16, 3).cuda()
        # pts   = pts / pts.norm(dim=1,keepdim=True)
        pts   = pts / pts.abs().sum(dim=1,keepdim=True)
        pts   = pts * torch.rand(pts.size(0),1,device=pts.device).sqrt() # Random inside sphere
        pts = pts * .5
        nrmls   = pts / pts.norm(dim=1,keepdim=True)
    else:
        ps     = torch.randn(512*2, 3).cuda()
        ps     = ps / ps.abs().sum(dim=1,keepdim=True)
        ps     = ps * torch.rand(ps.size(0),1,device=ps.device).sqrt() # Random inside sphere
        ps     = ps * .25
        nrmls0 = ps / ps.norm(dim=1,keepdim=True)
        ps0    = ps - .2

        ps1    = ps + .2
        nrmls1 = ps / ps.norm(dim=1,keepdim=True)
        pts    = torch.cat((ps0, ps1), 0)
        nrmls  = torch.cat((nrmls0, nrmls1), 0)
    print(' - pts  :\n', pts)
    print(' - nrmls:\n', nrmls)
    run_psr(pts, nrmls)
    # viz_conv_of_lap()
    # viz_box()
