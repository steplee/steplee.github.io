import torch, torch.nn.functional as F
import numpy as np, sys, time
torch.set_printoptions(linewidth=150, edgeitems=10)

from .solvers import solve_cg, solve_cg_AtA
from ..polygonalization.run_marching_algo import run_marching, show_marching_viz

from .singlegrid import get_stencil, get_laplacian, get_convolved_lap, get_grad, \
        forward_into_grid, backward_from_grid, \
        iter_stencil, iter_stencil_nonzero

from torch.utils.cpp_extension import load
extMod = load('extMod', sources=['poisson/psr/reduce.cu'], extra_cuda_cflags=['--extended-lambda', '-g'], extra_cflags=['-g'])

def test_extMod():
    torch.manual_seed(0)
    lvlF = 6
    lvlC = 3
    indsF = torch.randint(0, 1<<lvlF, (3,500)).cuda()
    # indsC = torch.randint(0, 1<<lvlC, (3,200)).cuda()
    indsC = indsF[:,:100] >> (lvlF - lvlC)
    # indsC,levelC = indsF,lvlF # WARNING: for same level test

    # We want them sorted properly...
    stF = torch.sparse_coo_tensor(indsF,torch.zeros_like(indsF[0])).coalesce()
    stC = torch.sparse_coo_tensor(indsC,torch.zeros_like(indsC[0])).coalesce()
    indsF = stF.indices()
    indsC = stC.indices()
    print(stF)
    print(stC)

    # gradStencil = get_grad(get_stencil()[2:3,2:3,2:3])
    gradStencil = get_grad(get_stencil())
    gradStencilSt = gradStencil.permute(1,2,3,0).to_sparse(3).cuda()
    print(get_stencil().shape)
    print(gradStencil.shape)
    print(gradStencilSt.shape)
    print(gradStencilSt.indices().shape)

    torch.cuda.synchronize()
    st = time.time()
    A = extMod.make_level_transfer(
            lvlF,lvlC,
            indsF,indsC,
            gradStencilSt
            )
    torch.cuda.synchronize()
    et = time.time()

    print(f' - n unique A[0]: {torch.unique(A.indices()[0]).numel()}')
    print(f' - n unique A[1]: {torch.unique(A.indices()[1]).numel()}')

    print(' - computed A:\n', A)
    print(f' - took {(et-st)*1000:.2f}ms')

    exit()
test_extMod()

class MultiGrid_Solver():
    def __init__(self,
            pnts, nrls
            ):
        self.pnts = pnts
        self.nrls = nrls

    def run_it(self, D=10):
        pts0 = self.pnts
        nrmls0 = self.nrls
        dev = pts0.device

        coo0, off, scale, size = forward_into_grid(pts0, D=D, pad=9)

        st0 = torch.sparse_coo_tensor(coo0.long().t(), nrmls0, size=(size,size,size,3)).coalesce()
        # st0._values(0.div_(st0._values().norm(dim=1,keepdim=True)))
        # NOTE: Consider not normalizing, that we we aren't merging bins and also losing samples.
        coo, nrmls2 = st0.indices(), F.normalize(st0.values(), dim=1)

        stencil = get_stencil()
        stencil_st = stencil.to_sparse().coalesce()
        gradStencil = get_grad(stencil)
        gradStencilSt = gradStencil.permute(1,2,3,0).to_sparse(3).cuda()

        # Compute vector field V
        # NOTE: We only ever need the divergence of V, can avoid computing it if needed.
        Vc,Vv = torch.empty((3,0),dtype=torch.long,device=dev), torch.empty((0,3),dtype=torch.float32,device=dev)
        for DD,W in iter_stencil(stencil_st):
            Vc = torch.cat((Vc, coo - DD.view(3,1)), 1)
            Vv = torch.cat((Vv, W*nrmls2), 0)
        V = torch.sparse_coo_tensor(Vc,Vv, size=(size,size,size,3)).coalesce()
        print(f' - V sizes (uncoalesced {Vc.size(1)}) (final nnz {V._nnz()}={V.indices().size(1)})')

        def get_v_for_lvl(V,d,D):
            V_ = V.coalesce()
            C = V_.values().size(-1) if V_.values().ndim == 2 else 1
            # print(f'{D} -> {d}')
            assert(d < D)
            assert(d > 0)
            '''
            for dd in range(D,d-1,-1):
                coos = V_.indices()
                print(coos[:,0])
                vals = V_.values()
                coos >>= 1
                size_ = 1 << d
                V_ = torch.sparse_coo_tensor(coos,vals, size=(size_,size_,size_,C)).coalesce()
            '''
            coos = V_.indices()
            # print('input coos\n', coos)
            vals = V_.values()
            coos = coos >> (D-d)
            size_ = 1 << (d)
            V_ = torch.sparse_coo_tensor(coos,vals, size=(size_,size_,size_,C)).coalesce()
            # print('output coos\n', V_.indices())

            return V_


        # Were going to have an Ax = b problem at each scale level.
        # Specifically, it will be Lx = v,
        # but we can use the least-squares form of conjugate gradients and instead use
        # the form J'Jx = v.
        # (J is the subselected matrix of function inner-products of gradient vector inner-products,
        #  and v is the divergence of V).
        #
        # Between scale levels, I did not consult any resources on multigrid techniques, so I
        # don't understand why it works, but Khazdan proposes transforming the target `b`/`v` vector
        # by subtracting portions of the solution at previous scale levels.
        # This is done with the A^fc_ij matrix. The ij-th element is like the laplacian between the i-basis node
        # on the fine grid and the j-th basis node on the coarse grid. It's not really a laplacian, but its form is similar,
        # it is the inner product of two gradients integrated over the support.

        # Run the process: solve each level from coarse to fine, and up-project the residual.
        # FIXME: Khazdan writes "Algorithm 1" in a way that requires Add' for each level to be kept in memory (or recomputed).
        #        Can't I instead keep all {bi} vectors in memory and subtract after each level solve?
        # No -> I don't think so, still need Add matrices at each iteration...
        #
        '''
        Khazdan has:
            for d in range(D):
                for d' in range(d):
                    bd = bd - Add' @ xd'
            Solve Ad @ xd = bd.

        But, can't I instead do:
            (initialize all b vectors)
            for d in range(D):
                Solve Ad @ xd = bd.
                for d' in range(d+1,D):
                    bd' = bd' - Ad'd @ xd

        NOTE: No: it still requires Ad'd at each level..
        '''

        # My first impl will use lots of ram...
        # NOTE: Do I downscale the computed-on-deepest-level-V for each level, OR do I downscale the raw
        #       data and recompute V on each level?
        minD = 1
        bs = {}
        solved_xs = {}
        level_transfers = {}

        '''
        for df in range(minD,D):
            for dc in range(minD, df):
                A_fc = extMod.make_level_transfer(df, dc, cooF, cooC, gradStencilSt)
                level_transfers[(df,dc)] = A_fc
        '''
        # for df in range(minD,D):
        for df in range(D,minD-1,-1):
            V_df = get_v_for_lvl(V,df,D) if df < D else V
            cooF = V_df.indices()
            for dc in range(minD, df):
                cooC = get_v_for_lvl(V,dc,D).indices()
                # cooC = get_v_for_lvl(V_df,dc,df).indices()
                A_fc = extMod.make_level_transfer(df, dc, cooF, cooC, gradStencilSt)
                print(f' - Computed lvl transfer {df:>2d} -> {dc:>2d} size {A_fc.shape} (original {cooF.shape} {cooC.shape})')
                level_transfers[(df,dc)] = A_fc

        for d in range(minD,D):
            print(f' - On level {d}')
            for dc in range(minD, d):
                # A_fc = make_level_transfer(d, dc, cooF, cooC, gradStencilSt)
                A_fc = level_transfers[(d,dc)]
                bd -= A_fc @ solved_xs[dc]
                print(f'\t- retract {dc} -> {d}')
            x = self.solve_level()
            dv = A_fc @ x
            Vd = get_v_for_lvl(V,d,D)
            print(f'\t- relax {d}, nnz(V)={Vd._nnz()}')



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
    '''
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
    '''

    mgs = MultiGrid_Solver(pts0, nrmls0)
    mgs.run_it(D);



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
    # print(' - pts  :\n', pts)
    # print(' - nrmls:\n', nrmls)
    run_psr(pts, nrmls)
    # viz_conv_of_lap()
    # viz_box()
