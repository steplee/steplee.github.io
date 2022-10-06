import torch, torch.nn, torch.nn.functional as F
import time, cv2
import sys

'''
This version supports different cameras seeing different subsets of points.
But They must all see the same number
'''


torch.manual_seed(0)
dev = torch.device('cpu')
# dev = torch.device('cuda')


PRINT_TIMER = False
def printTimer(st, name):
    if PRINT_TIMER:
        print(' - \'{:<12s}\' took {:.2f}ms'.format(name, 1000*(time.time()-st)))
def printElapsed(tt, name):
    if PRINT_TIMER:
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

    tpts =  (q_to_matrix(q).permute(0,2,1) @ (pts - t).permute(1,0,2).permute(0,2,1)).permute(0,2,1)
    # tpts =  (q_to_matrix(q).permute(0,2,1) @ (pts - t).permute(1,0,2).permute(0,2,1)).permute(0,2,1)
    return tpts[...,:2] / tpts[...,2:]

# Project N pts into one image
def mapProject2(pose, pts):
    q,t = pose[...,:4], pose[...,4:]

    # print(q_to_matrix1(q).T.shape,pts.shape, (pts-t.view(1,3)).T.shape)
    # tpts = (q_to_matrix1(q).T @ (pts - t.view(1,3)).T).T
    tpts = (pts-t.view(-1,3)) @ q_to_matrix1(q)
    return tpts[...,:2] / tpts[...,2:]


def generate_data(nposes, npts, nobs):
    pts = (torch.rand(npts,3) * 2 - 1) * 10
    pts[:,2] *= .1

    truePoses = torch.rand(nposes, 7)
    truePoses[:,:4] = torch.tensor([1.,0,0,0])
    truePoses[:, :4] += torch.randn_like(truePoses[:,:4]) * .0
    truePoses[:, :4]  = truePoses[:, :4] / truePoses[:, :4].norm(dim=-1,keepdim=True)
    truePoses[:,4:] -= .5
    truePoses[:,4:] *= 10
    truePoses[:,-1] *= .02
    truePoses[:,-1] -= 6 # they are in -Z axis

    predPoses = truePoses.clone()
    predPoses[:, :4] += (torch.rand_like(predPoses[:,:4]) - .5) * .2
    predPoses[:, :4]  = predPoses[:, :4] / predPoses[:, :4].norm(dim=-1,keepdim=True)
    predPoses[:, 4:] += (torch.rand_like(predPoses[:,4:]) -.5) * torch.tensor((.5,.5,.1)) * 2
    # print(predPoses)

    pts = pts.to(dev)
    truePoses = truePoses.to(dev)
    predPoses = predPoses.to(dev)
    observations_all = mapProject(truePoses, pts)

    # Each camera observed random subset of points
    obsPtIds = torch.stack([torch.randperm(npts) for _ in range(nposes)], 0)[:, :nobs]
    observations = observations_all.gather(1, torch.stack([obsPtIds]*2,-1))

    pts = pts + torch.randn_like(pts) * .01

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


class SparseOptimizer:
    def __init__(self):
        # nposes, npts, nobs = 2,6,5
        # nposes, npts, nobs = 100,1000,20
        # nposes, npts, nobs = 5,100,20
        # nposes, npts, nobs = 25,30,20
        nposes, npts, nobs = 50,200,20

        nstates = nposes*7 + npts*3
        pts, truePoses, predPoses, observations, obsPtIds = generate_data(nposes,npts,nobs)
        # observations = observations[...,:nobs, :]
        # print('obsPtIds',obsPtIds.shape)
        # print('pts',pts.shape)

        self.ptsForCamera = pts.view(1,npts,3).repeat(nposes,1,1).gather(1, torch.stack([obsPtIds]*3,-1))

        self.idxPose = lambda i,j: 7*i + j
        self.idxPt   = lambda i,j: 7*nposes + 3*i + j

        # ftMap, ftDMap = get_functorch_jac()
        self.ftMap2, self.ftDMap2 = get_functorch_jac_2()

        self.pts, self.truePoses, self.predPoses, self.observations, self.obsPtIds = \
                pts, truePoses, predPoses, observations, obsPtIds
        self.nposes, self.npts, self.nobs,self.nstates = nposes, npts, nobs, nstates
        self.iteration = 0

        self.mse = (self.ftMap2(predPoses, self.ptsForCamera) - observations).norm(dim=-1).mean()
        print('           - Initial mse', self.mse.item())


    def step(self):
        # Introduce locals
        pts, truePoses, predPoses, observations, obsPtIds = \
                self.pts, self.truePoses, self.predPoses, self.observations, self.obsPtIds
        ftMap2, ftDMap2 = self.ftMap2, self.ftDMap2
        ptsForCamera = self.ptsForCamera
        nposes, npts, nobs, nstates = self.nposes, self.npts, self.nobs, self.nstates

        t0 = time.time()

        # Form graph
        ind,val = [],[]
        resCnt = 0
        res = []

        tt = time.time()
        exp = ftMap2(predPoses, ptsForCamera)
        res = exp - observations
        js = ftDMap2(predPoses,ptsForCamera)
        printTimer(tt, 'AutoDiff')


        # print(' - *** res ***', res.shape)
        # print(' - *** JS  ***', [j.shape for j in js])

        res = [res.reshape(-1)]

        # NOTE: This is the bottleneck now, which is great because it is easily optimized
        tt = time.time()
        for posei in range(nposes):
            for obsi in range(nobs):
                J_pose = js[0][posei,obsi]
                J_pt = js[1][posei,obsi]
                # if posei == 0 and pti == 0: print('here',J_pose,'\n',J_pt)
                for ri in range(2):
                    for k in range(7):
                        ind.append((resCnt+ri, self.idxPose(posei,k)))
                        val.append(J_pose[ri,k])
                    for obsi in range(nobs):
                        pti = obsPtIds[posei, obsi]
                        for k in range(3):
                            ind.append((resCnt+ri, self.idxPt(pti,k)))
                            val.append(J_pt[ri,obsi,k])
                resCnt += 2

        # Add priors
        for i in range(nstates):
            '''
            if i < nposes*7:
                # Weak prior on pose translations, stronger on orientation
                val.append(.001 if i % 7 >= 4 else .1)
            else:
                # Strong prior on points
                val.append(.01)
            '''
            val.append(.8)
            ind.append((resCnt,i))
            resCnt += 1
        res.append(torch.zeros(nstates))
        printTimer(tt, 'GraphForm')

        res = torch.cat(res).view(-1)
        ind = torch.tensor(ind).T
        val = torch.tensor(val)

        tt = time.time()
        J = torch.sparse_coo_tensor(ind,val, size=(resCnt, nposes*7+npts*3))

        if J.shape[0] * J.shape[1] > 1e8:
            print(' - Refusing to operate on large matrix, until future update to use sparse matrix solve.')
            print(' - Matrix size:', J.shape)
            sys.exit(1)

        H = torch.sparse.mm(J.t(), J)
        g = J.t() @ res
        if self.iteration == 0:
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
        # print(' d shape',d.shape)
        # print(' d_pts shape',d_pts.shape)
        # print(' pts',pts.shape)

        for i,d_po in enumerate(d_poses):
            predPoses[i] -= d_po * 1
            predPoses[i,0:4] = F.normalize(predPoses[i,0:4], dim=0)
        for i,d_pt in enumerate(d_pts): pts[i] -= d_pt

        # print(' - took {:.5f}ms'.format((time.time()-t0)*1000))
        printTimer(t0,'Iteration')
        self.iteration += 1

        self.mse = (ftMap2(predPoses, ptsForCamera) - observations).norm(dim=-1).mean()
        print(' \t\t\t\t- Iteration',self.iteration,'mse',self.mse)


def do_sparse():
    sparse = SparseOptimizer()
    for iter in range(4):
        sparse.step()

def do_sparse_viz():
    from viz import Viz
    sparse = SparseOptimizer()

    # def set_data(self, pts, poses, fs, colors):
    vizPoseColors = torch.rand(sparse.nposes, 4).cuda()
    vizPoseColors[...,3] = .9
    vizPoseColors[...,:3] = F.normalize(vizPoseColors[...,:3])
    vizFs = torch.ones(sparse.nposes, 2).cuda()

    viz    = Viz()
    viz.set_data(sparse.pts.cuda(), sparse.predPoses.cuda(), vizFs, vizPoseColors)
    viz.wait = 1

    for i in range(9999):
        key,f = viz.draw()

        # if key == 'n':
        if True:
            cv2.imwrite('out/step_{:04d}.jpg'.format(sparse.iteration), f[...,[2,1,0]])
            sparse.step()
            viz.set_data(sparse.pts.cuda(), sparse.predPoses.cuda(), vizFs, vizPoseColors)



if 'viz' in sys.argv:
    do_sparse_viz()
else:
    do_sparse()
