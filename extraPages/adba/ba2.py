import torch, torch.nn, torch.nn.functional as F
import time
import sys

'''
This version supports different cameras seeing different subsets of points.
But They must all see the same number
'''


torch.manual_seed(0)
dev = torch.device('cpu')
# dev = torch.device('cuda')


def printTimer(st, name):
    print(' - \'{:<12s}\' took {:.2f}ms'.format(name, 1000*(time.time()-st)))
def printElapsed(tt, name):
    print(' - \'{:<12s}\' took {:.2f}ms'.format(name, 1000*(tt)))

def q_to_matrix1(q):
    # r,i,j,k = q[0], q[1], q[2], q[3]
    # return torch.stack((
    r,i,j,k = q[0:1], q[1:2], q[2:3], q[3:4]
    return torch.cat((
        1-2*(j*j+k*k), 2*(i*j-k*r), 2*(i*k+j*r),
        2*(i*j+k*r), 1-2*(i*i+k*k), 2*(j*k-i*r),
        2*(i*k-j*r), 2*(j*k+i*r), 1-2*(i*i+j*j))).view(3,3)
def q_to_matrix(q):
    r,i,j,k = q[:,0], q[:,1], q[:,2], q[:,3]
    return torch.stack((
        1-2*(j*j+k*k),
        2*(i*j-k*r),
        2*(i*k+j*r),
        2*(i*j+k*r),
        1-2*(i*i+k*k),
        2*(j*k-i*r),
        2*(i*k-j*r),
        2*(j*k+i*r),
        1-2*(i*i+j*j)),1).view(-1,3,3)

@torch.jit.script # <- Improves speed of 'jacobian' by about 30%
def mapProject(poses, pts):
    q, t = poses[:,:4], poses[:,4:]
    pts = pts.view(pts.size(0), 1,3)

    tpts =  (q_to_matrix(q) @ (pts - t).permute(1,0,2).permute(0,2,1)).permute(0,2,1)
    # tpts =  (q_to_matrix(q).permute(0,2,1) @ (pts - t).permute(1,0,2).permute(0,2,1)).permute(0,2,1)
    return tpts[...,:2] / tpts[...,2:]

# Project N pt into one image
def mapProject2(pose, pts):
    q,t = pose[...,:4], pose[...,4:]

    # print(q_to_matrix1(q).T.shape,pts.shape, (pts-t.view(1,3)).T.shape)
    # tpts = (q_to_matrix1(q).T @ (pts - t.view(1,3)).T).T
    tpts = (pts-t.view(-1,3)) @ q_to_matrix1(q)
    return tpts[...,:2] / tpts[...,2:]


def generate_data(nposes, npts, nobs):
    pts = torch.rand(npts,3) * 2 - 1

    truePoses = torch.rand(nposes, 7)
    truePoses[:,:4] = torch.tensor([1.,0,0,0])
    truePoses[:,4:] *= 2
    truePoses[:,4:] -= 1
    truePoses[:,-1] *= .1
    truePoses[:,-1] -= 2

    predPoses = truePoses.clone()
    predPoses[:, 4:] += torch.randn_like(predPoses[:,4:]) * torch.tensor((.5,.5,.1))

    pts = pts.to(dev)
    truePoses = truePoses.to(dev)
    predPoses = predPoses.to(dev)
    observations_all = mapProject(truePoses, pts)

    # Each camera observed random subset of points
    obsPtIds = torch.stack([torch.randperm(npts) for _ in range(nposes)], 0)[:, :nobs]
    observations = observations_all.gather(1, torch.stack([obsPtIds]*2,-1))

    return pts, truePoses, predPoses, observations, obsPtIds



#
# mapProject1 :: (H,7) x (H,3) -> (H,2)
#
#   mapProject2 :: (7)   x (H,3)   -> (H,2)
# v_mapProject2 :: (C,7) x (C,H,3) -> (C,H,2)
#
# (C,7) x (C,H,3) -> (C,H,2)
#
#
def get_functorch_jac_2():
    from functorch import jacfwd, vmap
    from functorch.compile import aot_function, memory_efficient_fusion, ts_compile

    # Each camera projects the SAME number of points

    m2 = vmap(mapProject2, (0,0))
    d_m2 = vmap(jacfwd(mapProject2, (0,1)), (0,0))

    return m2, d_m2



def do_sparse():
    # nposes, npts, nobs = 2,6,5
    nposes, npts, nobs = 100,1000,20

    nstates = nposes*7 + npts*3
    pts, truePoses, predPoses, observations, obsPtIds = generate_data(nposes,npts,nobs)
    # observations = observations[...,:nobs, :]
    print('obsPtIds',obsPtIds.shape)
    print('pts',pts.shape)

    ptsForCamera = pts.view(1,npts,3).repeat(nposes,1,1).gather(1, torch.stack([obsPtIds]*3,-1))

    idxPose = lambda i,j: 7*i + j
    idxPt   = lambda i,j: 7*nposes + 3*i + j

    # ftMap, ftDMap = get_functorch_jac()
    ftMap2, ftDMap2 = get_functorch_jac_2()

    for iter in range(4):
        t0 = time.time()

        # Form graph
        ind,val = [],[]
        resCnt = 0
        res = []
        mse = 0

        tt = time.time()
        # print('poses',predPoses.shape)
        # print('ptsForCamera',ptsForCamera.shape)
        exp = ftMap2(predPoses, ptsForCamera)
        # print('exp', exp.shape)
        # print('obs', observations.shape)
        res = exp - observations
        js = ftDMap2(predPoses,ptsForCamera)
        printTimer(tt, 'AutoDiff')

        # print(' - *** res ***', res.shape)
        # print(' - *** JS  ***', [j.shape for j in js])

        mse = res.norm(dim=-1).mean()
        res = [res.reshape(-1)]

        tt = time.time()
        for posei in range(nposes):
            for obsi in range(nobs):
                J_pose = js[0][posei,obsi]
                J_pt = js[1][posei,obsi]
                # if posei == 0 and pti == 0: print('here',J_pose,'\n',J_pt)
                for ri in range(2):
                    for k in range(7):
                        ind.append((resCnt+ri, idxPose(posei,k)))
                        val.append(J_pose[ri,k])
                    for obsi in range(nobs):
                        pti = obsPtIds[posei, obsi]
                        for k in range(3):
                            ind.append((resCnt+ri, idxPt(pti,k)))
                            val.append(J_pt[ri,obsi,k])
                resCnt += 2


        # Add priors
        for i in range(nstates):
            if i < nposes*7:
                # Weak prior on pose translations, stronger on orientation
                val.append(.001 if i % 7 >= 4 else .01)
            else:
                # Strong prior on points
                val.append(.01)
            ind.append((resCnt,i))
            resCnt += 1
        res.append(torch.zeros(nstates))
        printTimer(tt, 'GraphForm')

        res = torch.cat(res).view(-1)
        ind = torch.tensor(ind).T
        val = torch.tensor(val)
        print(res.shape,ind.shape)


        tt = time.time()
        J = torch.sparse_coo_tensor(ind,val, size=(resCnt, nposes*7+npts*3))

        if J.shape[0] * J.shape[1] > 1e8:
            print(' - Refusing to operate on large matrix, until future update to use sparse matrix solve.')
            print(' - Matrix size:', J.shape)
            sys.exit(1)

        H = torch.sparse.mm(J.t(), J)
        g = J.t() @ res
        if iter == 0:
            print(' - Sparsity stats')
            print(' \t- J ::',J.shape,'(nnz {}, {:.4f} occ)'.format(J._nnz(), J._nnz() / J.numel()))
            print(' \t- H ::',H.shape,'(nnz {}, {:.4f} occ)'.format(H._nnz(), H._nnz() / H.numel()))
        printTimer(tt, 'Sparse+Mult')

        tt = time.time()
        H = H.to_dense()
        printTimer(tt, 'Densify')
        tt = time.time()
        d = torch.linalg.solve(H, g)
        printTimer(tt, 'Solve')

        d_pts = d[nposes*7:].view(-1,3)
        d_poses = d[:nposes*7].view(-1,7)
        print(' d shape',d.shape)
        print(' d_pts shape',d_pts.shape)
        print(' pts',pts.shape)

        for i,d_po in enumerate(d_poses):
            predPoses[i] -= d_po * 1
            predPoses[i,0:4] = F.normalize(predPoses[i,0:4], dim=0)
        for i,d_pt in enumerate(d_pts): pts[i] -= d_pt

        # print(' - took {:.5f}ms'.format((time.time()-t0)*1000))
        printTimer(t0,'Iteration')
        print(' \t\t\t\t- Iteration',iter,'mse',mse)

do_sparse()
