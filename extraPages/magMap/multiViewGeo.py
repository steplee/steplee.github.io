import torch

def cross_matrix(k):
    if isinstance(k,torch.Tensor):
        return torch.cat((0, -k[2], k[1], k[2], 0, -k[0], -k[1], k[0], 0)).view(3,3)
    else:
        return torch.FloatTensor((0, -k[2], k[1], k[2], 0, -k[0], -k[1], k[0], 0)).view(3,3)

def homogeneous(a):
    out = torch.ones((*a.shape[:-1], a.shape[-1]+1))
    out[..., :a.shape[-1]] = a
    return out

def project(a):
    return a[...,:-1] / a[...,-1:]

def toMat4(A,t=None):
    B = torch.eye(4)
    if A.size() == (3,3):
        B[:3,3:] = R
        B[:3,3 ] = t
    else:
        B[:A.size(0),:A.size(1)] = A
        assert t is None
    return B

def invMat4(A):
    A = torch.eye(4)
    A[:3,3:] = A[:3,:3].T
    A[:3,3 ] = -A[:3,:3].T@A[:3,3]
    return A

def get_cam_from_essential(E):
    u,s,vt = torch.linalg.svd(E)
    print('singular vals of E should be nearly (1,1,0), they were', s)
    W = torch.FloatTensor((0,-1,0, 1,0,0, 0,0,1)).reshape(3,3)
    Z = torch.FloatTensor((0, 1,0,-1,0,0, 0,0,0)).reshape(3,3)
    R1 = U@W@V.T
    R2 = U@W.T@V.T
    S  = U@Z@U.T
    t  = torch.cat((S[2,1], S[0,2], S[1,0]))
    R = R1 if R1[2,2] > R2[2,2] else R2
    return R,t

def solve_essential(a,b):
    if a.size(-1) == 2: a = homogeneous(a)
    if b.size(-1) == 2: b = homogeneous(b)
    N,D = a.size()

    outer = (a.view(N,3,1) @ b.view(N,1,3)).view(N,9)
    # outer = (a.view(N,3,1) @ b.view(N,1,3)).permute(0,2,1).reshape(N,9)
    U,S,Vt = torch.linalg.svd(outer)

    # E1 = Vt[:,-1].view(3,3)
    E1 = Vt[-1].view(3,3)
    U2,S2,Vt2 = torch.linalg.svd(E1)
    # print(S2)
    S2[-1] = 0
    # A real essential matrix has 2 equal singular values and the last zero. But this makes my uncalibrated example haave lots of error.
    # S2[[0,1]] = torch.sqrt(S2[0]*S2[1])
    # print(S2)

    E = U2@torch.diag(S2)@Vt2
    E = E.mT

    # print('solve essential rmse:', (a * (b@E.T)).sum(1).pow(2).mean(0).sqrt().item())
    # print('solve essential rmse:', (b * (a@E.T)).sum(1).pow(2).mean(0).sqrt().item())
    print('solve essential rmse:', (b.view(N, 1, 3) @ E.view(1, 3, 3) @ a.view(N, 3, 1)).pow(2).mean(0).sqrt().item())
    return E

def solve_essential_ransac(ptsa, ptsb, threshold):
    assert(ptsa.shape[0] > 8)

    if ptsa.size(-1) == 2: ptsa = homogeneous(ptsa)
    if ptsb.size(-1) == 2: ptsb = homogeneous(ptsb)
    NN,D = ptsa.size()

    G = 10_000
    M = 8 # Appears to work with 7, but no less.

    # No batch-mode torch.randperm replacement. So I will do the slower thing and argsort random numbers
    # ids = torch.arange(NN, device=ptsa.device).view(1,N).repeat(G)
    ids = torch.rand(G,NN, device=ptsa.device).sort().indices[:, :M]

    a,b = ptsa[ids], ptsb[ids]

    outer = (a.view(G, M, 3, 1) @ b.view(G, M, 1, 3)).view(G, M, 9)
    U,S,Vt = torch.linalg.svd(outer)

    E1 = Vt[:, -1, :].view(G, 3, 3)
    U,S,Vt = torch.linalg.svd(E1)


    # S[..., -1] = 0
    SS = torch.zeros((G,3,3), device=S.device)
    SS[:,0,0] = S[:, 0]
    SS[:,1,1] = S[:, 1]
    SS[:,2,2] = 0
    # SS[:,2,2] = S[:, 2]
    E = U @ SS @ Vt
    E = E.permute(0, 2,1) # [G,3,3]

    # eval error.
    # From https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    blines = (E.view(G, 1, 3, 3) @ ptsa.view(1, NN, 3, 1)).view(G, NN, 3)
    alines = (E.view(G, 1, 3, 3).permute(0, 1, 3, 2) @ ptsb.view(1, NN, 3, 1)).view(G, NN, 3)
    errs  = (abs(ptsb.view(1, NN, 1, 3) @ blines.view(G,NN,3,1)).view(G,NN) / blines[:, :, :2].norm(dim=-1) \
          +  abs(ptsa.view(1, NN, 1, 3) @ alines.view(G,NN,3,1)).view(G,NN) / alines[:, :, :2].norm(dim=-1)) * .5

    valid = (errs.view(G, NN) < threshold).int()
    nvalid = valid.sum(1)
    best = nvalid.argmax()
    # print('best was', best, 'with nvalid', nvalid[best])
    # print(errs[best].squeeze())
    mask1 = valid[best]
    E1 = E[best]

    # Now solve again, one last time.
    E2 = solve_essential(ptsa[mask1>0], ptsb[mask1>0])
    blines = (E2.view(1, 3, 3) @ ptsa.view(NN, 3, 1)).view(NN, 3)
    alines = (E2.view(1, 3, 3).permute(0, 2, 1) @ ptsb.view(1, NN, 3, 1)).view(NN, 3)
    errs  = (abs(ptsb.view(NN, 1, 3) @ blines.view(NN,3,1)).view(NN) / blines[:, :2].norm(dim=-1) \
          +  abs(ptsa.view(NN, 1, 3) @ alines.view(NN,3,1)).view(NN) / alines[:, :2].norm(dim=-1)) * .5
    mask2 = (errs < threshold).int()
    print(' - solve essential inliers {}/{}/{}'.format(mask2.sum().item(), mask1.sum().item(), NN))

    return E1, E2, mask1, mask2

def triangulate_points(PA,PB,ptsa,ptsb):
    # Linear DLT method.
    if 1:
        # TODO: Normalization
        if ptsa.size(-1) == 2: ptsa = homogeneous(ptsa)
        if ptsb.size(-1) == 2: ptsb = homogeneous(ptsb)
        r1 = ptsa[:,0].view(-1,1) * PA[2].view(1,-1) - PA[0].view(1,-1)
        r2 = ptsa[:,1].view(-1,1) * PA[2].view(1,-1) - PA[1].view(1,-1)
        r3 = ptsb[:,0].view(-1,1) * PB[2].view(1,-1) - PB[0].view(1,-1)
        r4 = ptsb[:,1].view(-1,1) * PB[2].view(1,-1) - PB[1].view(1,-1)
        Ms = torch.stack((r1,r2,r3,r4),1)
        # print('Ms:\n',Ms)
        U,S,Vt = torch.linalg.svd(Ms)
        # pts = Vt[...,-1]
        pts = Vt[...,-1,:]
        pts = pts[:, :3] / pts[:, 3:]


    # Linear Inhomogeneous method.
    else:
        # TODO: Normalization
        if ptsa.size(-1) == 2: ptsa = homogeneous(ptsa)
        if ptsb.size(-1) == 2: ptsb = homogeneous(ptsb)
        r1 = ptsa[:,0] * PA[2] - PA[0]
        r2 = ptsa[:,1] * PA[2] - PA[1]
        r3 = ptsb[:,0] * PB[2] - PB[0]
        r4 = ptsb[:,1] * PB[2] - PB[1]
        Ms = torch.stack((r1,r2,r3,r4),1)[:,:3]
        U,S,Vt = torch.linalg.svd(Ms)
        n = ptsa.size(0)
        bs = torch.FloatTensor((0,0,0,1)).view(1,4).repeat(n,1)
        bprimes = (bs @ U) / S.view(n,-1)
        pts = bprimes @ Vt
        # FIXME: Untested.

    pptsa = project(homogeneous(pts) @ PA.T)[:,:2]
    pptsb = project(homogeneous(pts) @ PB.T)[:,:2]
    # print('pts:\n',pts)
    # print('ppts (A):\n',pptsa)
    # print('ppts (B):\n',pptsb)
    print('triangulated rmse in A:', ((pptsa - ptsa[:,:2])**2).sum(1).mean().sqrt().item())
    print('triangulated rmse in B:', ((pptsb - ptsb[:,:2])**2).sum(1).mean().sqrt().item())
    return pts

# The Lu algorithm
# Can definitely take a couple of iters if starting R matrix is not close.
# TODO: Take starting params.
# TODO: Batch and ransac-ify.
def solve_pnp(oo, p, K, R=None, t=None):
    o = homogeneous(oo)
    n = o.size(0)
    v = o @ torch.linalg.inv(K)
    V = (v.view(n, 3,1) @ v.view(n, 1,3) / v.pow(2).sum(1).view(n,1,1)).view(n,3,3)

    if R is None: R = torch.eye(3,device=oo.device)
    if t is None: t = torch.zeros(3,device=oo.device)

    e = (project((K.view(1,3,3) @ (R.view(1,3,3) @ p.view(n,3,1) + t.view(1,3,1)))[...,0]) - oo).pow(2).sum(-1).mean().sqrt().item()
    print(e)

    I = torch.eye(3).view(1,3,3)
    for iter in range(10):
        t = torch.linalg.inv(I[0] - V.mean(0)) @ ((V - I) @ R.view(1, 3, 3) @ p.view(n, 3, 1)).mean(0)

        q = (V @ (R.view(1,3,3) @ p.view(n,3,1) + t.view(1,3,1))).view(n,3)
        qp = q - q.mean(0,keepdim=True)
        pp = p - p.mean(0,keepdim=True)
        M = (qp.view(n,3,1) @ pp.view(n,1,3)).mean(0)
        U,S,Vt=torch.linalg.svd(M)
        R = Vt.mT @ U.mT

        e = (project((K.view(1,3,3) @ (R.view(1,3,3) @ p.view(n,3,1) + t.view(1,3,1)))[...,0]) - oo).pow(2).sum(-1).mean().sqrt().item()
        print(e)

def test_pnp():
    p = torch.FloatTensor((
        -1,-1, 1,
         1,-1, 1,
         1, 1, 1,
        -1, 1, 1,)).reshape(-1,3)
    o = p[:, :2] + .05
    K = torch.eye(3)
    solve_pnp(o,p,K)
# test_pnp(); exit()










def get_corners(s):
    return torch.FloatTensor((
        0,0,
        s,0,
        s,s,
        0,s)).view(-1,2)

def test_triang():
    w = 512
    K = torch.FloatTensor((w*.7, 0, 256, 0, w*.7, 256, 0, 0, 1)).view(3,3)
    A = torch.eye(4)[:3]
    B = torch.eye(4)[:3]; B[0,3] = .1
    ptsa = get_corners(w)
    ptsb = get_corners(w) + torch.FloatTensor((.8,0)).view(1,2)
    # ptsa,ptsb=ptsa[:1],ptsb[:1]
    triangulate_points(K@A, K@B, ptsa, ptsb)
test_triang()

def test_ess_1():
    w = 512

    K = torch.FloatTensor((w*.7, 0, 256, 0, w*.7, 256, 0, 0, 1)).view(3,3)
    Ki = torch.linalg.inv(K)
    ptsa = get_corners(w)
    ptsb = get_corners(w) + torch.FloatTensor((.8,0)).view(1,2)

    nptsa = ptsa @ Ki[:2,:2].T + Ki[:2,2]
    nptsb = ptsb @ Ki[:2,:2].T + Ki[:2,2]

    E = solve_essential(ptsa,ptsb)
    print(E)

def show_epipolar(imga,imgb,ptsa,ptsb,F):
    import cv2, numpy as np
    from matplotlib.cm import rainbow
    h,w = imga.shape[:2]
    dimg = cv2.cvtColor(np.hstack((imga,imgb)), cv2.COLOR_GRAY2BGR)
    cimg = dimg*0
    pimg = dimg*0
    limg = dimg*0

    # Draw matches.
    for a,b in zip(ptsa,ptsb):
        cv2.line(cimg, a, b+(w,0), (0,255,0), 1)

    # Draw epipolar lines.
    ptsa = np.hstack((ptsa, np.ones((ptsa.shape[0],1))))
    ptsb = np.hstack((ptsb, np.ones((ptsb.shape[0],1))))
    edgeLeft = np.cross(np.array((0,0,1),dtype=np.float32), np.array((0,h,1),dtype=np.float32))
    edgeRght = np.cross(np.array((w,0,1),dtype=np.float32), np.array((w,h,1),dtype=np.float32))
    print(edgeLeft, edgeRght)
    for i in range(len(ptsa)):
        c = tuple(int(255*c) for c in (rainbow(i/max(1,len(ptsa)-1))[:3]))[::-1]
        apt = ptsa[i]
        bline = F @ apt
        bleft = project(np.cross(bline, edgeLeft)).astype(int)
        bright = project(np.cross(bline, edgeRght)).astype(int)
        # print(bline, bleft, bright)
        cv2.line(limg, bleft+(w,0), bright+(w,0), c, 1)

        bpt = ptsb[i]
        aline = F.T @ bpt
        aleft = project(np.cross(aline, edgeLeft)).astype(int)
        aright = project(np.cross(aline, edgeRght)).astype(int)
        cv2.line(limg, aleft, aright, c, 1)

        cv2.circle(pimg, apt[:2].astype(int), 5, c, 1)
        cv2.circle(pimg, bpt[:2].astype(int)+(w,0), 5, c, 1)

    dimg = cv2.addWeighted(dimg, .99, cimg, .2, 0)
    dimg = cv2.addWeighted(dimg, .9, pimg, .997, 0)
    dimg = cv2.addWeighted(dimg, .9, limg, .9, 0)
    dimg = cv2.resize(dimg, (0,0), fx=.6, fy=.6)
    cv2.imshow('dimg', dimg)

def test_ess_2():
    import cv2, numpy as np
    imga = '/data/chromeDownloads/apt2/20230704_211407.jpg'
    imgb = '/data/chromeDownloads/apt2/20230704_211416.jpg'
    imga = cv2.imread(imga,0)
    imgb = cv2.imread(imgb,0)
    imga = cv2.resize(imga, (0,0), fx=.55, fy=.55)
    imgb = cv2.resize(imgb, (0,0), fx=.55, fy=.55)
    sift = cv2.SIFT_create(3000)
    kptsa,desa = sift.detectAndCompute(imga,None)
    kptsb,desb = sift.detectAndCompute(imgb,None)
    ptsa = np.stack([kpt.pt for kpt in kptsa],0)
    ptsb = np.stack([kpt.pt for kpt in kptsb],0)
    matcher = cv2.BFMatcher()
    ms = matcher.knnMatch(desa,desb,k=2)
    ms = [m for (m,mm) in ms if m.distance < mm.distance*.72]
    ptsa1 = ptsa[[m.queryIdx for m in ms]]
    ptsb1 = ptsb[[m.trainIdx for m in ms]]
    vfov = np.deg2rad(45)
    h,w = imga.shape[:2]
    K = np.array((
        (w/2)/np.tan(vfov/2), 0, w/2,
        0, (h/2)/np.tan(vfov/2), h/2,
        0, 0, 1),dtype=np.float32).reshape(3,3)
    E,mask = cv2.findEssentialMat(ptsa1, ptsb1, K, cv2.RANSAC, .99999, threshold=1.5)
    mask = mask.ravel()
    print(mask.sum(),'/',mask.size)
    ptsa2 = ptsa1[mask>0]
    ptsb2 = ptsb1[mask>0]
    show_epipolar(imga,imgb,ptsa2.astype(int),ptsb2.astype(int),np.linalg.inv(K.T)@E@np.linalg.inv(K))

    if 0:
        tK = torch.from_numpy(K)
        tKi = torch.linalg.inv(tK)
        t_ptsa3 = project(homogeneous(torch.from_numpy(ptsa2)) @ tKi.mT)
        t_ptsb3 = project(homogeneous(torch.from_numpy(ptsb2)) @ tKi.mT)
        tE = solve_essential(t_ptsa3, t_ptsb3)
        tE = tE.cpu().numpy()
        show_epipolar(imga,imgb,ptsa2.astype(int),ptsb2.astype(int),np.linalg.inv(K.T)@tE@np.linalg.inv(K))
    else:
        tK = torch.from_numpy(K)
        tKi = torch.linalg.inv(tK)
        t_ptsa3 = project(homogeneous(torch.from_numpy(ptsa1)) @ tKi.mT).cuda()
        t_ptsb3 = project(homogeneous(torch.from_numpy(ptsb1)) @ tKi.mT).cuda()
        # t_ptsa3 = project(homogeneous(torch.from_numpy(ptsa2)) @ tKi.mT) # WARNING:
        # t_ptsb3 = project(homogeneous(torch.from_numpy(ptsb2)) @ tKi.mT)
        # t_ptsa3,t_ptsb3 = t_ptsa3.cuda(), t_ptsb3.cuda()
        for i in range(10):
            threshold = tKi[0,0] * 10.9
            print('threshold',threshold)
            tE1, tE2, mask1, mask2 = solve_essential_ransac(t_ptsa3, t_ptsb3, threshold)
            tE = tE2.cpu().numpy()
            ptsa2 = ptsa1[mask2.cpu().numpy()>0]
            ptsb2 = ptsb1[mask2.cpu().numpy()>0]
            show_epipolar(imga,imgb,ptsa2.astype(int),ptsb2.astype(int),np.linalg.inv(K.T)@tE@np.linalg.inv(K))
            cv2.waitKey(0)





    cv2.waitKey(0)

with torch.no_grad():
    # test_ess_1()
    test_ess_2()
