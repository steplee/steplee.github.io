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

Using functorch is overkill: the jacobians are pretty trivial for SO3+Trans3...

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
    r,i,j,k = q[0:1], q[1:2], q[2:3], q[3:4]
    return torch.cat((
        1-2*(j*j+k*k), 2*(i*j-k*r), 2*(i*k+j*r),
        2*(i*j+k*r), 1-2*(i*i+k*k), 2*(j*k-i*r),
        2*(i*k-j*r), 2*(j*k+i*r), 1-2*(i*i+j*j))).view(3,3)

def q_mult1(p,q):
    a1,b1,c1,d1 = p
    a2,b2,c2,d2 = q
    return torch.FloatTensor((
        a1*a2 - b1*b2 - c1*c2 - d1*d2,
        a1*b2 + b1*a2 + c1*d2 - d1*c2,
        a1*c2 - b1*d2 + c1*a2 + d1*b2,
        a1*d2 + b1*c2 - c1*b2 + d1*a2))


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
            grad = torch.bmm(J.reshape(-1,2,7).mT, res.reshape(-1,2,1))[...,0].reshape(B,N,7).sum(1)
            JtJ = (J.permute(0,1,3,2) @ J).sum(1) # [B,7,7]

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

def cross_matrix(v):
    x,y,z = v.t()
    o = v[:,0]*0
    return torch.stack((
        o, -z, y,
        z, o, -x,
        -y, x, o), -1).view(v.size(0),3,3)


# Unlike above, here there is *ONE* pose for *ALL TIME*.
# It is shared, to find a static camera.
# def recover_camera_batch(cxy, fxy, obspts, worldpts,
def recover_camera_batch(intrins, obspts, worldpts,
                    initialEye=torch.FloatTensor((0,1,-2)),
                    initialQ=torch.FloatTensor((1,0,0,0)),
                    iters=20,
                    levenberg=True,
                    iniHessWeightR=10,
                    iniHessWeightT=10):

    B,N,two = obspts.size()
    # cxy = cxy.view(B,1,2).repeat(1,N,1).view(-1,2)
    # fxy = fxy.view(B,1,2).repeat(1,N,1).view(-1,2)
    cxy = torch.stack((intrins[:,0,2], intrins[:,1,2]), 1).view(B,1,2).repeat(1,N,1).view(-1,2)
    fxy = torch.stack((intrins[:,0,0], intrins[:,1,1]), 1).view(B,1,2).repeat(1,N,1).view(-1,2)
    print('cxy',cxy.shape,'obspts',obspts.shape)


    obspts = obspts.view(-1,2)
    worldpts = worldpts.view(-1,3)

    x = torch.cat((initialQ, initialEye), 0)
    n = worldpts.size(0)
    print(worldpts.size())

    levenberg_iters = 20 if levenberg else 1

    for i in range(iters):
        # Hess, grad = torch.eye(6), torch.zeros((6,))


        R = q_to_matrix1(x[:4])
        transpts = worldpts - x[4:]
        tpts = transpts @ R
        ppts = (tpts[:,:2] / tpts[:,2:]) * fxy + cxy
        res  = obspts - ppts
        rmse0 = res.pow(2).sum(1).mean().sqrt().item()
        print(f' - {i:>3d}: (rmse: {rmse0:.3f})')
        J_proj = torch.stack((
            fxy[:,0] / tpts[:,2], tpts[:,0]*0, -fxy[:,0]*tpts[:,0]/(tpts[:,2]*tpts[:,2]),
            tpts[:,0]*0, fxy[:,1] / tpts[:,2], -fxy[:,1]*tpts[:,1]/(tpts[:,2]*tpts[:,2])),1).view(-1,2,3)
        J_t = -R.mT.unsqueeze(0).repeat(n,1,1)
        J_r =  R.mT.unsqueeze(0) @ cross_matrix(transpts) @ R.unsqueeze(0)
        J = torch.cat((J_r,J_t), 2) # [N,3,6]
        J = J_proj @ J # [N,2,6]

        # Huber loss function
        eps = 35 # px
        Alpha = torch.eye(2).unsqueeze(0).repeat(n,1,1)
        err = res.norm(dim=1)
        mask = (err>eps).float()
        weight = mask*(eps/err) + (1-mask)
        # print(weight)
        # weight[:]=1
        Alpha *= weight.view(n,1,1)

        iniHess = torch.FloatTensor((
                    iniHessWeightR, iniHessWeightR, iniHessWeightR,
                    iniHessWeightT, iniHessWeightT, iniHessWeightT))

        #
        # LM: If the step would increase the error,
        #     increase prior weight of previous estimate by increasing a diagonal term added to hessian.
        #     This reduces the effect of the approx hessian and reduces the step size, making it more like gradient descent.
        #
        for j in range(levenberg_iters):

            grad = (J.permute(0,2,1) @Alpha@ res.view(n,2,1))[...,0] # [N,6]
            Hess = (J.permute(0,2,1) @Alpha@ J) # [N,6,6]

            grad = grad.sum(0)
            Hess = Hess.sum(0) + torch.diag(iniHess)

            dx = torch.linalg.solve(Hess,grad)
            # dx[:3] = 0

            x1 = x.clone()
            x1[:4] = q_mult1(x1[:4], q_exp(dx[:3]))
            x1[:4] = x1[:4] / x1[:4].norm()
            x1[4:] = x1[4:] + dx[3:]

            if levenberg:
                R1 = q_to_matrix1(x1[:4])
                transpts = worldpts - x1[4:]
                tpts = transpts @ R1
                ppts = (tpts[:,:2] / tpts[:,2:]) * fxy + cxy
                res1  = obspts - ppts
                rmse1 = res1.pow(2).sum(1).mean().sqrt().item()
                print(f' - {i}:{j}: compare rmse1 {rmse1:.2f} vs rmse0 {rmse0:.2f}')

                if rmse1 < rmse0:
                    # This iteration succeeded.
                    # print('step with', dx)
                    x = x1
                    break
                else:
                    # This iteration failed.
                    if j == levenberg_iters-1:
                        print(f' - Failed LM step at outer-iter {i}', dx)
                        return x
                    # iniHess *= 4 if j < 5 else 7
                    iniHess *= j+2 # Very fast growing.
            else:
                # Unconditionally accept
                x = x1

    return x


# Set implementation
# recover_camera = recover_camera_opencv
recover_camera = recover_camera_functorch



if __name__ == '__main__':
    torch.manual_seed(57)
    # torch.manual_seed(5)
    wh = torch.FloatTensor((512,512.))
    uv = np.tan(np.deg2rad(50)/2)*2
    fxy = wh/uv

    N = 10
    # worldpts = torch.randn(1,N,3) + torch.FloatTensor((0,1,8)).view(1,1,3)
    worldpts = torch.randn(1,N,3)# + torch.FloatTensor((0,1,8)).view(1,1,3)

    # pose0 = torch.FloatTensor((1,0,.3,-.02, 0,0.1,-6.02)).view(7)
    pose0 = torch.FloatTensor((1,0,.6,-.02, 0,0.1,-4.02)).view(7)
    pose0[...,:4] = nn.functional.normalize(pose0[...,:4], dim=-1)


    intrins = torch.eye(3)
    intrins[0,0] = fxy[0]
    intrins[1,1] = fxy[1]
    intrins[0,2] = wh[0]*.5
    intrins[1,2] = wh[1]*.5
    intrins = intrins.view(1,3,3).repeat(worldpts.size(0),1,1)

    # obspts = mapProject2(pose0, worldpts, fxy, wh)
    R = q_to_matrix1(pose0[:4]).view(1,3,3)
    obspts = (R.permute(0,2,1) @ (worldpts - pose0[4:].view(1,1,3)).permute(0,2,1))
    print(intrins.shape,obspts.shape)
    obspts = (intrins @ obspts).permute(0,2,1)
    obspts = obspts[...,:2] / obspts[...,2:]
    # print(obspts)
    obspts += torch.randn_like(obspts) * .0001

    x=recover_camera_batch(intrins, obspts, worldpts)
    print(pose0)
    print(x)

    exit()


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

