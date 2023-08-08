import torch, torch.nn as nn, torch.nn.functional as F, numpy as np


'''

Two implementations, chosen at bottom of file.

functorch based PNP solver.
Requires a decent initial estimate.

Actually opencv solvePnP is not very good unless you use the iterative method with an initial estimate,
which is about as good as my NLLS solver. IIRC solvePnP use Lu OI algorithm though, which converges fast, probably
unlike the NLLS solve.
Without useExtrinsicGuess, I figure solvePnP either does OI with a bad initial scene plane, or attempts to first use an algebraic
method like EPNP and simply fails.

'''


def log_R(R):
    t = np.arccos((np.trace(R) - 1) * .5)
    d = np.linalg.norm((R[2,1]-R[1,2], R[0,1]-R[1,0], R[0,2]-R[2,0]))
    if d < 1e-12: return np.zeros(3)
    return np.array((R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1])) * t / d
def q_exp(r):
    l2 = r@r
    if l2 < 1e-15:
        return np.array((1,0,0,0.))
    l = np.sqrt(l2)
    # a = l * np.pi * .5
    a = l * .5
    c,s = np.cos(a), np.sin(a)
    # return np.array((0,1,0,0))
    return np.array((c,*((s/l)*r)))

def q_to_matrix1(q):
    # r,i,j,k = q[0], q[1], q[2], q[3]
    # return torch.stack((
    r,i,j,k = q[0:1], q[1:2], q[2:3], q[3:4]
    return torch.cat((
        1-2*(j*j+k*k), 2*(i*j-k*r), 2*(i*k+j*r),
        2*(i*j+k*r), 1-2*(i*i+k*k), 2*(j*k-i*r),
        2*(i*k-j*r), 2*(j*k+i*r), 1-2*(i*i+j*j))).view(3,3)

def mapProject2(pose, pts, fxy, wh):
    q,t = pose[...,:4], pose[...,4:]
    tpts = (pts-t.view(-1,3)) @ q_to_matrix1(q) # Transposing R twice: R.T.T = R
    ppts = (tpts[...,:2] / tpts[...,2:])
    # ppts[...,1] *= -1
    return ppts * fxy + wh*.5

def get_functorch_jac_2():
    from functorch import jacfwd, vmap
    from functorch.compile import aot_function, memory_efficient_fusion, ts_compile

    # Each camera projects the SAME number of points
    m2 = vmap(mapProject2, (0,0,0,0))
    d_m2 = vmap(jacfwd(mapProject2, (0,1)), (0,0,0,0))

    return m2, d_m2

# Actually this is not very good.
def recover_camera_opencv(wh, fxy, obspts, worldpts,
                   initialEye=torch.FloatTensor((0,1,-2)),
                   initialQ=torch.FloatTensor((1,0,0,0))):

    rvec0 = log_R(q_to_matrix1(initialQ))
    tvec0 = -(q_to_matrix1(initialQ) @ initialEye).cpu().numpy()


    import cv2
    Ps = []
    for i in range(len(obspts)):
        opts = worldpts[i].cpu().numpy()
        ipts = obspts[i].cpu().numpy()
        K = np.array((
            fxy[0], 0, wh[0]*.5,
            0, fxy[1], wh[1]*.5,
            0,0,1)).reshape(3,3)
        print(opts.shape, ipts.shape, K.shape)
        if 0:
            stat,rvec,tvec = cv2.solvePnP(opts,ipts,K,None)
        else:
            stat,rvec,tvec = cv2.solvePnP(opts,ipts,K,None, np.copy(rvec0),np.copy(tvec0), useExtrinsicGuess=True)
        assert(stat)
        P = np.eye(4)
        # print('K',K)
        rod = cv2.Rodrigues(rvec)
        print('tvec',tvec.squeeze())
        print('rvec',rvec)
        print('R',rod[0])
        # P[:3,:3] = rod[0]
        # P[:3,3] = tvec.squeeze()
        P = np.zeros(7)
        if 1:
            # Must invert
            P[:4] = q_exp(rvec.squeeze()) * (1,-1,-1,-1)
            P[4:] = -(rod[0].T @ tvec.squeeze())
        else:
            P[:4] = q_exp(rvec.squeeze())# * (1,-1,-1,-1)
            P[4:] = tvec.squeeze()
        # print('R_',q_to_matrix1(torch.from_numpy(P[:4])))
        Ps.append(P)
    return torch.from_numpy(np.stack(Ps))


def recover_camera_functorch(wh, fxy, obspts, worldpts,
                   initialEye=torch.FloatTensor((0,1,-2)),
                   initialQ=torch.FloatTensor((1,0,0,0)),
            ):

    B,N,three = worldpts.size()
    x = torch.cat((initialQ,initialEye), -1).view(1,7).repeat(B,1)
    x[...,:4] = nn.functional.normalize(x[...,:4], dim=-1)
    wh = wh.view(1,2).repeat(B,1)
    fxy = fxy.view(1,2).repeat(B,1)

    Nstate = 7
    Nobs = N * 2

    prior0 = torch.eye(7).unsqueeze_(0)
    prior0[:,:4,:4] *= 1e4
    prior0[:,4:,4:] *= 1e2
    prior = prior0.clone()

    F, dF = get_functorch_jac_2()

    # print('obs\n',obspts)

    for i in range(32):
        pred = F(x, worldpts, fxy, wh)
        Js = dF(x, worldpts, fxy, wh)


        # print('pred\n',pred)
        res = pred - obspts # [B,N,2]
        rmse = res.pow(2).sum(2).mean(1).sqrt() # [B]
        print(f' - step {i} rmse {rmse}')

        # print(pred.shape,[JJ.shape for JJ in Js])
        if 0:
            J = Js[0].sum(1) # sum out 'N', the observation dim. to get a [B,2,7] tensor.

            # grad = (J.mT @ res.mT).sum(2) # [B,2,7] x [B,N,2] -> [B,7]
            grad = (J.mT @ res.permute(0,2,1)).sum(2) # [B,2,7] x [B,N,2] -> [B,7]

            # JtJ = J.mT @ J # [B,7,7]
            JtJ = J.permute(0,2,1) @ J # [B,7,7]
        else:
            J = Js[0] # [B,N,2,7]
            # print('j',J.shape, 'res',res.shape)
            # print((J.mT @ res.permute(0,2,1).unsqueeze(-2)).shape)
            # grad = (J.mT @ res.permute(0,2,1)).sum(1)
            # print(torch.bmm(J.view(-1,2,7).mT, res.view(-1,2,1))[...,0]
            # print(J.shape, res.shape)
            grad = torch.bmm(J.reshape(-1,2,7).mT, res.reshape(-1,2,1))[...,0].reshape(B,N,7).sum(1)
            JtJ = (J.permute(0,1,3,2) @ J).sum(1) # [B,7,7]
            # print(J.shape,grad.shape,JtJ.shape)

        # print('JtJ:\n',JtJ)
        # print('grad:',grad)
        if 1:
            Hess = JtJ + prior
            P = Hess.inverse()
            # print('P:\n',P)

            bad = (P.diagonal(dim1=1,dim2=2) <= 0)
            if (bad).any():
                print(P.diagonal(dim1=1,dim2=2))
                print(' - WARNING: non-pos-definite covariance.')
                # prior = prior*2

                # continue
                # P[bad.flatten(1).any(1)] = 0
                # exit()

            # print(P.shape,grad.shape)
            x = x - (P @ grad.unsqueeze(-1))[...,0]
        else:
            Hess = JtJ[...,4:,4:] + prior[...,4:,4:]
            P = Hess.inverse()
            # print('P',P.shape)
            x[...,4:] = x[...,4:] - (P @ grad[...,4:].unsqueeze(-1))[...,0]

        x[...,:4] = nn.functional.normalize(x[...,:4], dim=-1)
        # x[...,:4] = torch.FloatTensor((1,0,0,0)).view(1,4)
        # print('q',x[0,:4], 't',x[0,4:])
    print('q',x[0,:4], 't',x[0,4:])

    return x




# Set implementation
# recover_camera = recover_camera_opencv
recover_camera = recover_camera_functorch






if __name__ == '__main__':
    wh = torch.FloatTensor((512,512.))
    uv = np.tan(np.deg2rad(50)/2)*2
    fxy = wh/uv
    worldpts = torch.randn(1,10,3) + torch.FloatTensor((0,1,8)).view(1,1,3)

    # def mapProject2(pose, pts, fxy, wh):
    # obspts = (worldpts * 1)[:,:,:2] * fxy + wh*.5
    pose0 = torch.FloatTensor((1,0,.1,-.02, 0,2.1,-1.02)).view(7)
    pose0[...,:4] = nn.functional.normalize(pose0[...,:4], dim=-1)
    obspts = mapProject2(pose0, worldpts, fxy, wh)

    recover_camera(wh, fxy, obspts, worldpts)
    # exit()

