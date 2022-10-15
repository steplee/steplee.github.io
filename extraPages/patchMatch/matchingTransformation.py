import numpy as np
np.random.seed(0)

def isometry_inv_3(A):
    out = np.eye(4, dtype=A.dtype)
    out[:3,:3] = A[:3,:3].T
    out[:3, 3] = -A[:3,:3].T @ A[:3,3]
    return out

def homogoneousize(x):
    if x.ndim == 1:
        return np.array((*x,1))
    return np.concatenate((x, np.ones_like(x[...,-1:])), -1)

def normalize(x):
    return x / np.linalg.norm(x,axis=-1,keepdims=True)

def compute_normalizer_2d(pa):
    t = pa.mean(0)
    s = np.sqrt(2) / np.sqrt((pa*pa).sum(1).mean())
    T = np.eye(3)
    T[:2,2] = -t * s
    T[0,0] = T[1,1] = s
    return T

'''
Zisserman-Hartley, Chapter 11
Normalized 8-point algorithm

# TODO: Implement either the iterative Gold-Standard LS optim, or the iterative algebraic refinement.
'''
def fundamental_from_pts(pa,pb):
    N = len(pa)
    assert N >= 8

    Ta = compute_normalizer_2d(pa)
    Tb = compute_normalizer_2d(pb)
    pa = pa @ Ta[:2,:2] + Ta[:2,2]
    pb = pb @ Tb[:2,:2] + Tb[:2,2]

    A = np.zeros((N,9), dtype=np.float64)
    for i, ((x,y),(u,v)) in enumerate(zip(pa,pb)):
        A[i] = u*x, u*y, u, v*x, v*y, v, x, y, 1

    u,s,vt = np.linalg.svd(A)
    F1 = vt[-1].reshape(3,3)

    u,s,vt = np.linalg.svd(F1)
    s[2] = 0
    F = (u*s)@vt

    F = Tb.T @ F @ Ta

    return F

def cross_matrix(k):
    return np.array((
        0,-k[2], k[1],
        k[2], 0, -k[0],
        -k[1], k[0], 0)).reshape(3,3)

def epipoles_from_fundamental(F):
    u,s,vt = np.linalg.svd(F)
    a = u[:, -1]
    b = vt[-1]
    a,b = b,a
    # print(' e\' . F\' . e`')
    # print((a.T) @ F.T @ (b))
    return a,b # is this correct?

'''
Zisserman-Hartley, Section 9.2.2
'''
def fundamental_from_matrices(PA, PB, KA, KB):
    X = KA @ (PB[:3]) @ isometry_inv_3(PA)

    Xplus = np.zeros((4,3), dtype=PA.dtype)
    Xplus[:3,:3] = np.linalg.inv(KA)
    C = np.array((0,0,0,1),dtype=PA.dtype)

    F = cross_matrix(X @ C) @ X @ Xplus
    return F

def project(x):
    return x[...,:2] / x[...,2:]

def rodrigues(w):
    t = np.linalg.norm(w) + 1e-12
    k = w / t
    return np.eye(3) + np.sin(t) * cross_matrix(w) + (1-np.cos(t)) * cross_matrix(w) @ cross_matrix(w)

# Test that some random points projected satisfy the fundamental matrix constraints
def test_F_from_matrices():
    PA = np.eye(4)
    PB = np.eye(4)
    KA,KB = np.eye(3), np.eye(3)
    PA[:3,:3] = rodrigues(np.array((.0,0,0)))
    PB[:3,:3] = rodrigues(np.array((.0,0,.4)))
    PA[:3,3] = .1,0,-1
    PB[:3,3] = 0,.1,-1

    wpts = np.random.randn(10,3)
    apts = project((wpts @ PA[:3,:3].T) + PA[:3,3].T) @ KA[:2,:2].T + KA[:2,2].T
    bpts = project((wpts @ PB[:3,:3].T) + PB[:3,3].T) @ KB[:2,:2].T + KB[:2,2].T

    F = fundamental_from_matrices(PA,PB, KA,KB)
    print(F)
    print(' epi error')
    for w,a,b in zip(wpts,apts,bpts):
        print(homogoneousize(a) @ F.T @ homogoneousize(b))


def test_F_from_pts():
    pts1 = np.random.randn(10,2)
    true_pts2 = pts1 + (.1,.2)
    true_pts2 = true_pts2 @ np.array((np.cos(1), -np.sin(1), np.sin(1), np.cos(1))).reshape(2,2)
    obs_pts2 = true_pts2 + np.random.randn(*true_pts2.shape) * .01

    F = fundamental_from_pts(pts1, obs_pts2)
    print('F:\n',F)
    print(' - epi errors, based on noisy observations:')
    for i in range(len(pts1)):
        print(homogoneousize(pts1[i]) @ F.T @ homogoneousize(true_pts2[i]), (homogoneousize(pts1[i]) @ F.T @ homogoneousize(obs_pts2[i])))

    print(' - epi from F :: ',epipoles_from_fundamental(F))


# https://ethaneade.com/rot_between_vectors.pdf
def rot_a_into_b(a,b):
    a,b = normalize(a), normalize(b)
    w = np.cross(a,b)
    K = cross_matrix(w)
    return np.eye(3) + K + (1 / (1 + a@b)) * K@K

'''
Zisserman-Hartley, Sections 11.12.1 and 11.12.2
'''
# def find_matching_transformation(PA, PB, KA, KB, ptsa, ptsb):
def find_matching_transformation(F, ptsa, ptsb):
    N = len(ptsa)
    ea, eb = epipoles_from_fundamental(F)

    # Step 1: Find HB
    if 1:
        x0 = np.zeros(2) # Chosen central point with minimal distortion
        # x0 = ptsb.mean(0)
        T = np.eye(3)
        T[:2,2] = -x0

        # TODO: If angle is less when rotating other way do that (do not require f be positive)
        ee = eb
        # ee = ea
        ang = np.arctan2(ee[1], ee[0])
        R = np.eye(3)
        R[0,0] = R[1,1] = np.cos(ang)
        R[1,0] = -np.sin(ang)
        R[0,1] = np.sin(ang)
        moved_epipole = R@ee
        moved_epipole /= moved_epipole[2]
        f = moved_epipole[0]
        assert abs(moved_epipole[1]) < 1e-10, moved_epipole
        print('f',f, R@ee, R@T@ee)

        G = np.eye(3)
        G[2,0] = -1/f

        HB = G @ R @ T
        # HB = np.linalg.inv(T) @ G @ R @ T
        print('HB:\n',HB)
        print('HB @ ee:',HB@ee)


    # Step 2: Find HA = HAA @ HB @ M
    if 1:
        if 1:
            # TODO: IS THIS M?
            # F = cross(e) @ M
            # M = np.linalg.pinv(cross_matrix(ee)) @ F
            # u,s,vt = np.linalg.svd(M)
            # print(s)
            # M = M + np.outer(vt[-1],vt[-1])
            # M = M + np.outer(u[:,-1], u[:,-1])
            # assert np.allclose(F, cross_matrix(ee)@M)
            # M = M/M[2,2]


            # K = np.eye(3)
            # K[:2,2] = -ptsa.mean()
            # F = K @ F @ K
            u,s,vt = np.linalg.svd(F)
            print(s)
            w = np.array((0,-1,0,1,0,0,0,0,1.)).reshape(3,3)
            z = np.array((0,1,0, -1,0,0, 0,0,0.)).reshape(3,3)
            t = u[:,2]
            M = u@w@vt
            # M = cross_matrix(t)@u@w@vt
            # M = u@vt
            '''
            v = np.array((.0001,0,0.))
            # v = np.array((.000001,0,0.))
            v = np.array((.00010,0,0.))
            M = cross_matrix(ee) @ F + np.outer(ee,v)
            '''

            # M = np.eye(3)
            print('ERR',F - cross_matrix(ee)@M)
        else:
            import cv2
            M,_,_ = cv2.decomposeEssentialMat(F)


        H0 = HB @ M
        # H0 = M @ HB
        # H0 /= H0[2,2]
        print('M:\n',M, 'ee',ee)
        print('H0:\n',H0, 'ee',ee)

        ahat = project(homogoneousize(ptsa) @ H0.T)
        bhat = project(homogoneousize(ptsb) @ HB.T)
        print('first 10 ahats and bhats:\n', np.concatenate((ahat,bhat),-1)[:10])

        A = np.zeros((N,3))
        res = np.zeros(N)
        for i in range(N):
            A[i] = ahat[i,0], ahat[i,1], 1
            # res[i] = ahat[i,0] - bhat[i,0]
            res[i] =   bhat[i,0]
        # print('A',A.T)
        # abc = np.linalg.solve(A.T @ A, A.T @ res)
        prob = np.linalg.lstsq(A, res)
        abc = prob[0]
        print(np.linalg.svd(A.T@A))
        print(' - solved abc', abc)
        print(' - lstsq res ', prob[1])

        HAA = np.eye(3)
        # HAA[0] = abc # WARNING: Disabled

        HA = HAA @ H0
        HA /= HA[2,2]

        print(' - pts              ', abs(ptsa[:,0]-ptsb[:,0]).sum())
        print(' - original residual', abs(res).sum())
        # print(' - post+projresidual', abs(project(homogoneousize(ptsa)@(HA))[:,0] - project(bhat)[:,0]).sum())
        print(' - post     residual', abs(project(homogoneousize(ptsa)@(HA.T))[:,0] - bhat[:,0]).sum())

        print(' - pts               y', abs(ptsa[:,1]-ptsb[:,1]).sum())
        print(' - pre +projresidual y', abs(project(homogoneousize(ptsa)@(H0.T))[:,1] - (bhat)[:,1]).sum())
        print(' - post+projresidual y', abs(project(homogoneousize(ptsa)@(HA.T))[:,1] - (bhat)[:,1]).sum())

    print('h0\n',H0)
    print('ha\n',HA)
    print('hb\n',HB)
    return HA, HB

def test_matching_xform():
    pts1 = np.random.randn(10,2)
    true_pts2 = pts1 + (.05,.1)
    true_pts2 = true_pts2 @ np.array((np.cos(.01), -np.sin(.01), np.sin(.01), np.cos(.01))).reshape(2,2)

    F = fundamental_from_pts(pts1, true_pts2)

    find_matching_transformation(F, pts1, true_pts2)

def test_matching_xform_real_data():
    import cv2
    # imga,imgb = cv2.imread('data/dino4.jpg'), cv2.imread('data/dino3.jpg')
    # imga,imgb = cv2.imread('data/mike1.jpg'), cv2.imread('data/mike2.jpg')
    imgb,imga = cv2.imread('data/strathmore1.jpg'), cv2.imread('data/strathmore2.jpg')
    imga = cv2.resize(imga, (0,0), fx=.5, fy=.5)
    imgb = cv2.resize(imgb, (0,0), fx=.5, fy=.5)
    sift = cv2.SIFT_create(2500)
    kptsa, desa = sift.detectAndCompute(imga,None)
    kptsb, desb = sift.detectAndCompute(imgb,None)
    ptsa = np.array([kpt.pt for kpt in kptsa])
    ptsb = np.array([kpt.pt for kpt in kptsb])
    matcher = cv2.BFMatcher()
    mms = matcher.knnMatch(desa,desb,k=2)
    ms = [m1 for m1,m2 in mms if m1.distance < m2.distance * .75]
    pts1a = ptsa[[m.queryIdx for m in ms]]
    pts1b = ptsb[[m.trainIdx for m in ms]]
    Fcv,mask = cv2.findFundamentalMat(pts1a, pts1b, cv2.FM_RANSAC, 3)
    mask = mask.squeeze()
    print('matches', len(mms))
    print('lowe matches', len(ms))
    print('epipolar matches', mask.sum())
    pts2a = pts1a[mask==1]
    pts2b = pts1b[mask==1]

    # Debug matching
    if 1:
        baseImg, scale = np.concatenate((imga,imgb), 1), 1
        baseImg, scale = cv2.resize(baseImg, (0,0), fx=.5, fy=.5), scale * .5
        w = baseImg.shape[1]//2
        for pt in ptsa: cv2.circle(baseImg, (  int(scale*pt[0]), int(scale*pt[1])), 1, (0,255,0), -1)
        for pt in ptsb: cv2.circle(baseImg, (w+int(scale*pt[0]), int(scale*pt[1])), 1, (0,255,0), -1)
        for pta, ptb in zip(pts1a, pts1b):
            cv2.line(baseImg,
                    (int(scale*pta[0]), int(scale*pta[1])),
                    (int(w+scale*ptb[0]), int(scale*ptb[1])), (120,2,200), 1)
        for pta, ptb in zip(pts2a, pts2b):
            cv2.line(baseImg,
                    (int(scale*pta[0]), int(scale*pta[1])),
                    (int(w+scale*ptb[0]), int(scale*ptb[1])), (202,255,0), 1)
        dimg = baseImg
        cv2.imshow('dimg', dimg)
        cv2.waitKey(0)

    # pts2a = pts2a - pts2a.mean(0)
    # pts2b = pts2b - pts2b.mean(0)
    F = fundamental_from_pts(pts2a, pts2b)
    # F=Fcv
    # ea,eb = epipoles_from_fundamental(Fcv)
    # print(' - Epipole A (normalized):', ea/ea[2])
    # print(' - Epipole B (normalized):', eb/eb[2])
    ea,eb = epipoles_from_fundamental(F)
    print(' - Epipole A (normalized):', ea/ea[2])
    print(' - Epipole B (normalized):', eb/eb[2])
    Ha, Hb = find_matching_transformation(F, pts2a, pts2b)
    h,w = imga.shape[:2]
    ra = cv2.warpPerspective(imga,Ha, (w,h))
    rb = cv2.warpPerspective(imgb,Hb, (w,h))
    dimg = np.concatenate((ra,rb), 1)
    dimg = cv2.resize(dimg, (0,0), fx=.5, fy=.5)
    cv2.imshow('dimg', dimg)
    cv2.waitKey(0)



# test_F_from_matrices()
# test_F_from_pts()
# test_matching_xform()
test_matching_xform_real_data()
