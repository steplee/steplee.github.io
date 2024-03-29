import numpy as np, cv2, os, sys
import torch
from matplotlib.cm import rainbow

from .solve import *
from .render import *
sys.path.append(os.getcwd() + '/../')
from LoFTR_copied.run import run_pair

def get_frames(file, stride=30,N=4):
# def get_frames(file, stride=18,N=4):
    good = True
    frames = []

    if file.endswith('.mp4'):
        vcap = cv2.VideoCapture(file)

        for i in range(99999):
            good,frame = vcap.read()
            if not good: break
            if i % stride == 0:

                if 0:
                    # frame = cv2.resize(frame, (0,0), fx=.25,fy=.25)
                    frameg = cv2.GaussianBlur(frame, (0,0), 3)
                    frame = cv2.addWeighted(frame, 1.5, frameg, -.5, 0)
                    frameg = cv2.GaussianBlur(frame, (0,0), 3)
                    frame = cv2.addWeighted(frame, 1.5, frameg, -.5, 0)

                frames.append(frame)
                if len(frames) >= N: break

    else:
        files = sorted(os.listdir(file))[:N]
        for f in files:
            f = os.path.join(file,f)
            frames.append(cv2.imread(f))

    return frames

def extract(frames):
    kptss,dess = [],[]
    if 0:
        sift = cv2.SIFT_create(4000)
        # sift = cv2.SIFT_create(1200)
        # sift = cv2.SIFT_create(5200)
        # sift = cv2.SIFT_create(5200)
        # sift = cv2.AKAZE_create(3, threshold=.001)
        # sift = cv2.AKAZE_create(3, threshold=.005)
        for i,frame in enumerate(frames):
            kpts,des = sift.detectAndCompute(frame,mask=None)
            dimg = np.copy(frame,'C')
            for kpt in kpts:
                cv2.circle(dimg, (int(kpt.pt[0]),int(kpt.pt[1])), 2, (0,255,0), 1)
            if dimg.shape[0] > 1400: dimg = cv2.resize(dimg, (dimg.shape[1]*1400//dimg.shape[0],1400))
            cv2.imshow('dimg',dimg)
            cv2.waitKey(1)

            kpts = np.array([[kpt.pt[0],kpt.pt[1],kpt.size,kpt.angle] for kpt in kpts]).astype(np.float32)
            kptss.append(kpts)
            dess.append(des)
    elif 0:
        from .superpoint import get_sp, extract_sp
        sp = get_sp('/data/saves/superpoint_v1.pth')

        for i,frame in enumerate(frames):
            if frame.shape[-1] == 3: frame_ = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
            else: frame_ = frame
            kpts,des = extract_sp(sp,frame_)
            dimg = np.copy(frame,'C')
            for kpt in kpts:
                cv2.circle(dimg, (int(kpt[0]),int(kpt[1])), 2, (0,255,0), 1)
            if dimg.shape[0] > 1400: dimg = cv2.resize(dimg, (dimg.shape[1]*1400//dimg.shape[0],1400))
            cv2.imshow('dimg',dimg)
            cv2.waitKey(1)

            kpts = np.array([[kpt[0],kpt[1]] for kpt in kpts]).astype(np.float32)
            kptss.append(kpts)
            dess.append(des)

    elif 1:
        # from torchvision.models import resnet50
        from .myResnet import resnet50
        m = resnet50(pretrained=True).cuda().eval()

        D = 16 # Cell size

        for i,frame in enumerate(frames):
            scale = 1
            mh = 1400
            mh = 600
            if frame.shape[0] > mh:
                scale = mh/frame.shape[0]
                frame = cv2.resize(frame, (frame.shape[1]*mh//frame.shape[0],mh))
            h,w = frame.shape[:2]

            # if frame.shape[-1] == 3: frame_ = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
            Y = torch.arange(0,h,D).cuda()
            X = torch.arange(0,w,D).cuda()
            kpts = torch.cartesian_prod(Y,X).flip(1).float() * (1./scale)
            xx = torch.from_numpy(frame).cuda().float()
            x = (xx.div_(255).unsqueeze_(0).permute(0,3,1,2) - .4) / .23
            f = m(x)[0].permute(1,2,0)
            des = f.view(-1,f.size(2))

            xxx = torch.zeros((
                ((h+D-1)//D) * D,
                ((w+D-1)//D) * D,
                3), device=xx.device)
            xxx[0:xx.size(1), 0:xx.size(2)] = xx[0]
            print(xxx.shape, xxx.shape[0]//D, xxx.shape[1]//D)
            STD_THRESHOLD = .006
            STD_THRESHOLD = .01
            mask = (xxx.mean(2).view(xxx.shape[0]//D,D,xxx.shape[1]//D,D).permute(0,2,1,3).reshape(xxx.shape[0]//D,xxx.shape[1]//D,-1).std(2) > STD_THRESHOLD)
            mask = mask[:Y.shape[0], :X.shape[0]]
            mask[0:2] = False
            mask[-2:] = False
            mask[:,0:2] = False
            mask[:,-2:] = False
            mask = mask.reshape(-1)

            kpts,des = kpts[mask], des[mask]
            kptss.append(kpts.cpu().numpy())
            dess.append(des.cpu().numpy())
            print(kpts.shape,des.shape)
    else:
        pass


    return kptss,dess

def get_intrin(vfov, wh):
    v2 = np.tan(np.deg2rad(vfov)*.5)
    f = np.array((wh[1]*.5/v2, wh[1]*.5/v2))
    c = np.array((wh[0]/2,wh[1]/2))
    K = np.array((
        f[0],0,c[0],
        0,f[1],c[1],
        0,0,1)).astype(np.float32).reshape(3,3)

    return f,c,K # lol.

# NUMPY

def q_to_rot(q):
    q0,q1,q2,q3 = q
    return torch.FloatTensor((
        q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2-q0*q3), 2*(q0*q2+q1*q3),
        2*(q1*q2+q0*q3), (q0*q0-q1*q1+q2*q2-q3*q3), 2*(q2*q3-q0*q1),
        2*(q1*q3-q0*q2), 2*(q0*q1+q2*q3), q0*q0-q1*q1-q2*q2+q3*q3)).view(3,3)
def q_mult(p,q):
    a1,b1,c1,d1 = p
    a2,b2,c2,d2 = q
    if isinstance(p,torch.Tensor):
        return torch.FloatTensor((
            a1*a2 - b1*b2 - c1*c2 - d1*d2,
            a1*b2 + b1*a2 + c1*d2 - d1*c2,
            a1*c2 - b1*d2 + c1*a2 + d1*b2,
            a1*d2 + b1*c2 - c1*b2 + d1*a2))
    return np.array((
        a1*a2 - b1*b2 - c1*c2 - d1*d2,
        a1*b2 + b1*a2 + c1*d2 - d1*c2,
        a1*c2 - b1*d2 + c1*a2 + d1*b2,
        a1*d2 + b1*c2 - c1*b2 + d1*a2))
def q_exp(r):
    assert r.size(-1) == 3
    a2 = r@r
    if a2 < 1e-10: return torch.FloatTensor((1,0,0,0))
    a = np.sqrt(a2)
    k = r / a
    ha = a * .5
    return torch.FloatTensor((np.cos(ha), *(k*np.sin(ha))))

def createPoseFromMat4(P):
    if isinstance(P,torch.Tensor): P = P.cpu().numpy()
    w = np.sqrt(1 + P[0,0] + P[1,1] + P[2,2]) * .5
    w = max(w, 1e-7)
    x = (P[2,1] - P[1,2]) / (4*w)
    y = (P[0,2] - P[2,0]) / (4*w)
    z = (P[1,0] - P[0,1]) / (4*w)
    n = np.sqrt(w**2+x**2+y**2+z**2)
    w,x,y,z = w/n,x/n,y/n,z/n
    t = P[:3,3]
    return torch.FloatTensor((
        t[0],t[1],t[2],w,x,y,z))
def createPose(rvec,tvec,invert):
    rvec = torch.from_numpy(rvec).float()
    tvec = torch.from_numpy(tvec).float()
    print('WARNING: make sure this is correct and compatiable with cv2')
    if invert:
        qc = q_exp(-rvec)
        # print(q_to_rot(qc))
        # print(np.linalg.det(q_to_rot(qc)))
        return torch.cat((q_to_rot(qc) @ -tvec, qc))
    else:
        return torch.cat((tvec, q_exp(rvec)))

def homogeneous(a):
    return np.concatenate((a,np.ones_like(a[:,-1:])), -1)
def project(a):
    return a[...,:-1] / a[...,-1:]

# TORCH

def crossMatrix(k):
    return torch.FloatTensor((0, -k[2], k[1],   k[2], 0, -k[0], -k[1], k[0], 0)).reshape(3,3)

def box_plus(x,d):
    y = x.clone()
    y[:3] += d[:3]
    y[3:] = q_mult(y[3:], q_exp(d[3:]))
    return y

def get_rmse(a):
    if isinstance(a,torch.Tensor):
        return (a.mT@a).mean().sqrt()
    else:
        return np.sqrt((a.T@a).mean())

# OPENCV

def decomposeEssential(E,K,pts1,pts2):
    '''
    R1,R2,t = cv2.decomposeEssentialMat(E)
    t = t.reshape(-1,1)

    if R2[2,2] > R1[2,2]:
        R = R2
    else:
        R = R1

    # R=R1
    # T = np.concatenate((R, -t), 1)
    # T = np.concatenate((R.T, -R@t), 1)
    T = np.concatenate((R, t), 1)
    T = np.concatenate((T, np.array((0,0,0,1),dtype=T.dtype).reshape(1,4)),0)
    T = np.linalg.inv(T)
    print('using B matrix from E as\n',T)
    return T
    '''

    # pts1 = homogeneous(pts1) @ np.linalg.inv(K).T
    # pts2 = homogeneous(pts2) @ np.linalg.inv(K).T
    # pts1 = pts1[:,:2] / pts1[:,2:]
    # pts2 = pts2[:,:2] / pts2[:,2:]
    # stat,R,t,mask = cv2.recoverPose(E, pts1, pts2)
    # stat,R,t,mask = cv2.recoverPose(E, pts1, pts2)
    stat,R,t,mask = cv2.recoverPose(E, pts1, pts2, cameraMatrix=K)
    # stat,R,t,mask = cv2.recoverPose(E, pts1, pts2, cameraMatrix=np.linalg.inv(K))
    print('recoverMask ratio:', (mask>0).astype(np.float32).mean())
    # print(R,t,mask)
    T = np.concatenate((R, t), 1)
    T = np.concatenate((T, np.array((0,0,0,1),dtype=T.dtype).reshape(1,4)),0)
    T = np.linalg.inv(T)
    print('using B matrix from E as\n',T)
    return T

    '''
    Ts = (
        np.concatenate((R1, t), 1),
        np.concatenate((R1, -t), 1),
        np.concatenate((R2, t), 1),
        np.concatenate((R2, -t), 1) )

    for T in Ts:
        # P = K @ T
        P = np.zeros((3,4))
        P[:3,:3] = T[:3,:3].T
        P[:3,3] = -T[:3,:3].T @ T[:3,3]
        print(T.shape, homogeneous(pts2).shape)
        # ppts2 = (T @ homogeneous(pts2)).T
        ppts2 = homogeneous(pts2) @ T
        if (ppts2[:,2] > 0).all():
            return T

    print('*********************************')
    print('decomposeEssential FAILED')
    print('E=\n',E)
    print('R1=\n',R1)
    print('R2=\n',R2)
    print('t=',t)
    print('pts1=\n',pts1)
    print('pts2=\n',pts2)
    assert False
    '''

def triangulateRelative(K,E,pts1,pts2):
    B0 = decomposeEssential(E,K,pts1,pts2)
    PA = K @ np.eye(4)[:3]
    PB = K @ np.linalg.inv(B0)[:3]
    # PB = K @ (B0)[:3]

    pts4 = cv2.triangulatePoints(PA,PB,pts1.T,pts2.T).T
    pts = pts4[:,:3] / (pts4[:,3:])

    print(' - B0 is:\n', B0)
    print(' - Reprojected error triangulated points in A are:', get_rmse(pts1-project((PA@homogeneous(pts).T).T)))
    print(' - Reprojected error triangulated points in B are:', get_rmse(pts2-project((PB@homogeneous(pts).T).T)))
    print(' - Fraction of points behind camera in A are:', ((PA@homogeneous(pts).T).T[:,2] < 0).astype(np.float32).mean())
    print(' - Fraction of points behind camera in B are:', ((PB@homogeneous(pts).T).T[:,2] < 0).astype(np.float32).mean())
    print(' - Fraction of points too-close camera A are:', ((PA@homogeneous(pts).T).T[:,2] < 1e-1).astype(np.float32).mean())
    print(' - Fraction of points too-close camera B are:', ((PB@homogeneous(pts).T).T[:,2] < 1e-1).astype(np.float32).mean())
    # print(pts)
    return pts, B0

def triangulateAbsolute(A,B,K,pts1,pts2):

    if 1:

        AA = np.linalg.inv(A)
        BB = np.linalg.inv(B)

        CC = np.eye(4,dtype=A.dtype)
        DD = BB @ np.linalg.inv(AA)

        PC = K @ CC[:3]
        PD = K @ DD[:3]

        pts4 = cv2.triangulatePoints(PC,PD,pts1.T,pts2.T).T
        pts = pts4[:,:3] / pts4[:,3:]

        pts = (A[:3,:3]@pts.T).T + A[:3,3]

        PA = K @ AA[:3]
        PB = K @ BB[:3]
    else:
        PA = K @ np.linalg.inv(A)[:3]
        PB = K @ np.linalg.inv(B)[:3]
        pts4 = cv2.triangulatePoints(PA,PB,pts1.T,pts2.T).T
        pts = pts4[:,:3] / pts4[:,3:]

    print(' - Reprojected error triangulated points in A are:', get_rmse(pts1-project((PA@homogeneous(pts).T).T)))
    print(' - Reprojected error triangulated points in B are:', get_rmse(pts2-project((PB@homogeneous(pts).T).T)))
    print(' - Fraction of points behind camera in A are:', ((PA@homogeneous(pts).T).T[:,2] < 0).astype(np.float32).mean())
    print(' - Fraction of points behind camera in B are:', ((PB@homogeneous(pts).T).T[:,2] < 0).astype(np.float32).mean())
    print(' - Fraction of points too-close camera A are:', ((PA@homogeneous(pts).T).T[:,2] < 1e-1).astype(np.float32).mean())
    print(' - Fraction of points too-close camera B are:', ((PB@homogeneous(pts).T).T[:,2] < 1e-1).astype(np.float32).mean())

    # print(' - Reprojected error triangulated points in A are:\n', pts1-project((K@A[:3]@homogeneous(pts).T).T))
    # print(' - Reprojected error triangulated points in B are:\n', pts2-project((K@B[:3]@homogeneous(pts).T).T))

    return pts

class Runner:
    def __init__(self, file):

        self.file = file
        self.frames = get_frames(file)

        self.f,self.c,self.K = get_intrin(31.5, self.frames[0].shape[:2][::-1])
        print(self.f,self.K)
        self.wh=self.c*2

        self.kptss = [None for _ in range(len(self.frames))]
        self.frameGraph = self.matchLinear_loftr()
        self.kptss = [k.cpu().numpy().astype(np.float32) for k in self.kptss]

        # first_H_info = self.find_largest_homog() # This'll dictate the "ground" plane.
        first_H_info = None
        poseGraph = self.construct_graph(self.frameGraph, first_H_info)


    # It may make more sense to do this dynamically, while adding new frames and extending the pose graph.
    # That's how large SfM does it.
    # But our linear/chain assumption helps.
    def associate_matched(self, linearFrameGraph):

        frameAndIdx2id = {}

        # Initial frame has all new keypoint ids.
        for i in range(len(self.kptss[0])):
            frameAndIdx2id[(0,i)] = len(frameAndIdx2id)

        # This is kind of like a disjoint set data structure
        for entry in linearFrameGraph:
            preFrameId = entry['a']
            curFrameId = entry['b']
            idxsa = entry['verifiedIdsa']
            idxsb = entry['verifiedIdsb']

            # We may have missed some; do prev image first
            for (idxa,idxb) in zip(idxsa,idxsb):
                if (preFrameId,idxa) not in frameAndIdx2id:
                    frameAndIdx2id[(preFrameId,idxa)] = len(frameAndIdx2id)


            for (idxa,idxb) in zip(idxsa,idxsb):
                if (preFrameId,idxa) in frameAndIdx2id:
                    frameAndIdx2id[(curFrameId,idxb)] = frameAndIdx2id[(preFrameId,idxa)]
                else:
                    frameAndIdx2id[(curFrameId,idxb)] = len(frameAndIdx2id)

        return frameAndIdx2id



    def construct_graph(self, linearFrameGraph, first_H_info):
        G = []
        # cstCamCam = []
        # cstCamKpt = []

        # 'fi2id' = frameAndPtIdx to graph id
        fi2id = self.associate_matched(linearFrameGraph)

        camStateSize = 3 + 4
        camTangentSize = 3 + 3
        NC = len(self.frames) # Num cameras
        NK = len(fi2id) # Num keypoints

        for entry in linearFrameGraph: NK += len(entry['verifiedIdsa'])
        NS = NC*camStateSize + 3*NK # Total state size
        NT = NC*camTangentSize + 3*NK # Total tangent size
        x = torch.zeros((NS),dtype=torch.float32)
        b = np.zeros((NT),dtype=np.float32)

        camStateId = {i:i*camStateSize for i in range(NC)}
        kptStateId = {i:(camStateSize*NC)+i*3 for i in range(NK)}
        camTangentId = {i:i*camTangentSize for i in range(NC)}
        kptTangentId = {i:(camTangentSize*NC)+i*3 for i in range(NK)}

        def fetchStatePts(ids):
            out = torch.zeros((len(ids),3),dtype=x.dtype)
            for i,id in enumerate(ids):
                offset = kptStateId[id]
                out[i] = x[offset:offset+3]
            return out
        def fetchFrameIdPts(frame,idsInFrame):
            ids = [fi2id[(frame,id)] for id in idsInFrame]
            return fetchStatePts(ids)
        def getCamAsMatrix(i):
            s = x[camStateId[i]:camStateId[i]+camStateSize]
            t = s[:3]
            q = s[3:7]
            # print('cam',i,'t',t,'q',q)
            A = torch.eye(4)
            A[:3,:3] = q_to_rot(q)
            A[:3,3] = t
            return A


        # The first camera has a unary constraint to be fixed at t=(0,0,0), q=(1,0,0,0)
        # So the first 6 elements in the graph will be a diag matrix with target b vector = 0
        for i in range(6):
            G.extend([i, i, 100])

        # Assume first_H_info is about 0<->1. So we do that edge up front.
        # Triangulate the points and initialize them. Then solvePnP for the second camera. Repeat.
        E01 = linearFrameGraph[0]['E']
        pts01a,pts01b,idsa,idsb = self.getVerifiedMatchingPts(0,1)
        origPts, secondPose = triangulateRelative(self.K, E01, pts01a, pts01b)
        origPts = torch.from_numpy(origPts)
        for i,origPt in enumerate(origPts):
            x[kptStateId[idsa[i]]:kptStateId[idsa[i]]+3] = origPt

        # Set initial two cameras
        poseIdentity = torch.FloatTensor((0,0,0, 1,0,0,0))
        x[camStateId[0]:camStateId[0]+camStateSize] = poseIdentity
        x[camStateId[1]:camStateId[1]+camStateSize] = createPoseFromMat4(secondPose)
        print('Converted vs original:')
        print(getCamAsMatrix(1))
        print(secondPose)

        # Let's represent graph as sparse coo matrix.
        # It corresponds to the stacked jacobian matrices of the constraints.
        # We'll then convert to CSR or whatever later for the cg solver.
        #
        # Any sparse matrix has 3 values per entry: the row, column, and value.
        # We can build the *structure* now. The only thing that will change over the course of the optimization
        # are the *values*.
        #
        # Here I build with a python list of triplets, then swizzle later as needed
        iniPoseConstraints = 6
        unaryCoo = [
            0,0,9900,
            1,1,9900,
            2,2,9900,
            3,3,9915,
            4,4,9915,
            5,5,9915]
        numResiduals = iniPoseConstraints

        numLandmarkObs = sum([len(entry['verifiedIdsa']) for entry in linearFrameGraph])
        numLandmarkConstraints = (6+3) * numLandmarkObs
        landmarkCoo = []

        # Each row will have 9 non-zeros columns that needn't be tracked: 6 will be at camTangentId[fi]:+6 and 3 will be at kptTangentId[ki]:+3
        # Each keypoint obs creates two rows (for x and y observations)
        # This is helpful to keep track of the rows.
        fiToLandmarkRow = {}
        for entry in linearFrameGraph:
            a,b = entry['a'],entry['b']
            imgIdsa = entry['verifiedIdsa']
            imgIdsb = entry['verifiedIdsb']

            for imgId in imgIdsa:
                fiToLandmarkRow[(a,imgId)] = numResiduals
                fi = camTangentId[a]
                ki = kptTangentId[fi2id[(a,imgId)]]
                for i in range(6): landmarkCoo.extend((numResiduals  , fi+i, 1e-6))
                for i in range(3): landmarkCoo.extend((numResiduals  , ki+i, 1e-6))
                for i in range(6): landmarkCoo.extend((numResiduals+1, fi+i, 1e-6))
                for i in range(3): landmarkCoo.extend((numResiduals+1, ki+i, 1e-6))
                numResiduals += 2

            for imgId in imgIdsb:
                fiToLandmarkRow[(b,imgId)] = numResiduals
                for i in range(6): landmarkCoo.extend((numResiduals  , fi+i, 1e-6))
                for i in range(3): landmarkCoo.extend((numResiduals  , ki+i, 1e-6))
                for i in range(6): landmarkCoo.extend((numResiduals+1, fi+i, 1e-6))
                for i in range(3): landmarkCoo.extend((numResiduals+1, ki+i, 1e-6))
                numResiduals += 2

        if 0:
            coo = np.concatenate((
                unaryCoo,
                np.array(landmarkCoo).reshape(-1,3)
            ), 0)
            print(' - Precomputed structure shape (COO) : ', coo.shape)
        else:
            coo = unaryCoo + landmarkCoo
            a = coo[0::3]
            b = coo[1::3]
            c = coo[2::3]
            coo = torch.stack((torch.LongTensor(a),torch.LongTensor(b)),0)
            val = torch.FloatTensor(c)
            st = torch.sparse_coo_tensor(coo,val).coalesce()
            print(' - Sparse Matrix info:\n', st)

        worldPtsSeenByEachCamera = [[]]*NC


        for entry in linearFrameGraph:
            a,b = entry['a'],entry['b']
            imgIdsa = entry['verifiedIdsa']
            imgIdsb = entry['verifiedIdsb']
            worldPtIdsa = [fi2id[(a,id)] for id in imgIdsa]
            worldPtIdsb = [fi2id[(a,id)] for id in imgIdsa]
            worldPtsSeenByEachCamera[a].extend(worldPtIdsa)
            worldPtsSeenByEachCamera[b].extend(worldPtIdsb)
        worldPtsSeenByEachCamera = [np.unique(np.array(a).astype(np.uint32)) for a in worldPtsSeenByEachCamera]

        if 1:
            cameras = x[:NC*camStateSize]
            cameras = torch.stack([getCamAsMatrix(i) for i in range(NC)], 0).cpu().numpy()
            worldPts = x[NC*camStateSize:].cpu().numpy()
            self.update_viz(
                    cameras,
                    worldPts,
                    worldPtsSeenByEachCamera)


        # Run loop: add all points and cameras
        # Each iter:
        #              0) Pose A is gauranteed to be initialized, and there must be some A/B matches.
        #              1) Initialize B pose with solvePnP on _already_ triangulated points.
        #              2) Triangulate all of the new points that are not yet existing.
        #              3) Add blocks to graph structure
        for entry in linearFrameGraph:
            a,b = entry['a'],entry['b']
            # if entry['a'] == 0 and entry['b'] == 1: continue

            imgIdsa = entry['verifiedIdsa']
            imgIdsb = entry['verifiedIdsb']
            obsPtsa = self.kptss[a][imgIdsa][:,:2]
            obsPtsb = self.kptss[b][imgIdsb][:,:2]
            statePtsa = fetchFrameIdPts(a, imgIdsa)
            statePtsb = fetchFrameIdPts(b, imgIdsb)
            existingMask = (statePtsa != 0).all(1)
            print(f' - Adding Matched Image {a} <-> {b}')
            print(f'\t- Have {existingMask.sum()}/{statePtsa.shape[0]} existing matches to pnp off of')

            # print('statePtsA:',statePtsa.shape,'\n',statePtsa)
            # print('statePtsB:',statePtsb.shape,'\n',statePtsb)
            # print('existingPtsA:',statePtsa[existingMask].shape,'\n',statePtsa[existingMask])
            # print('existingPtsB:',statePtsb[existingMask].shape,'\n',statePtsb[existingMask])

            pnpPtsObjb = statePtsb[existingMask]
            pnpPtsObsb = obsPtsb[existingMask]

            stat,rvec,tvec = cv2.solvePnP(pnpPtsObjb.cpu().numpy(), pnpPtsObsb, self.K, None)
            rvec,tvec = rvec.reshape(-1), tvec.reshape(-1)
            # R = cv2.Rodrigues(rvec)[0]
            x[camStateId[b]:camStateId[b]+camStateSize] = createPose(rvec,tvec,invert=True)
            # x[camStateId[b]:camStateId[b]+camStateSize] = createPose(rvec,tvec,invert=False)

            # print(stat,rvec,tvec)
            # print(getCamAsMatrix(1))
            # print(secondPose)

            if 1:
                A = getCamAsMatrix(a)
                B = getCamAsMatrix(b)

                AA = np.linalg.inv(A)
                # print('AA:\n',AA)
                pnpPtsObja = statePtsa[existingMask]
                pnpPtsObsa = obsPtsa[existingMask]
                tstPts = (self.K @ AA[:3] @ homogeneous(pnpPtsObja).T).T
                tstPts = tstPts[:,:2] / tstPts[:,2:]
                print('Projected error (px) of A pose with old points:',get_rmse(tstPts - pnpPtsObsa))

                BB = np.linalg.inv(B)
                # BB = B
                # print('BB:\n',BB)
                tstPts = (self.K @ BB[:3] @ homogeneous(pnpPtsObjb).T).T
                tstPts = tstPts[:,:2] / tstPts[:,2:]
                print('Projected error (px) of new pose with pnp points:',get_rmse(tstPts - pnpPtsObsb))
                # print('Projected error (px) of new pose with pnp points:',(tstPts - pnpPtsObsb))

            # Now triangulate any un-initialized points
            if (~existingMask).sum() > 0:
                createImgPtsa = obsPtsa[~existingMask]
                createImgPtsb = obsPtsb[~existingMask]
                print(f' - Triangulating {len(createImgPtsa)} new points')
                newPts = triangulateAbsolute(A.cpu().numpy(),B.cpu().numpy(),self.K, createImgPtsa, createImgPtsb)
                newPts = torch.from_numpy(newPts)
                # statePtsa[~existingMask] = newPts
                # print('new points are:\n', statePtsa[~existingMask])
                newIdsa = imgIdsa[existingMask==0]
                for i,id in enumerate([fi2id[(a,id)] for id in newIdsa]):
                    x[kptStateId[id]:kptStateId[id]+3] = newPts[i]

            assert((statePtsa != 0).all(1).any())

        if 1:
            cameras = x[:NC*camStateSize]
            cameras = torch.stack([getCamAsMatrix(i) for i in range(NC)], 0).cpu().numpy()
            worldPts = x[NC*camStateSize:].cpu().numpy()
            self.update_viz(
                    cameras,
                    worldPts,
                    worldPtsSeenByEachCamera)

        # TODO: Implement optimization loop with CG and repeated re-linearizations
        fx,fy = self.f
        torchK = torch.from_numpy(self.K)
        eps = 10
        eps2 = eps*eps
        # st[0,0]=1 # no can do ;(
        for i in range(32):

            print('iter',i)

            # fiToLandmarkRow = {}
            newCoo = []
            newRes = []
            newAlpha = []
            newNumResiduals = 0
            for frameId_imgPtId, landmarkRow in fiToLandmarkRow.items():
                fi,ki = frameId_imgPtId
                # Batch this! Damn, R is even shared!

                wStateOff = kptStateId[fi2id[frameId_imgPtId]]
                pqStateOff = camStateId[fi]
                wTanOff = kptTangentId[fi2id[frameId_imgPtId]]
                pqTanOff = camTangentId[fi]

                w = x[wStateOff:wStateOff+3]

                tq = x[pqStateOff:pqStateOff+7]
                t = tq[:3]
                q = tq[3:7]
                R = q_to_rot(q)

                xform_pt = R.T @ (w-t)
                proj_pt0 = torchK @ xform_pt
                proj_pt = proj_pt0[:2] / proj_pt0[2:]
                obs = self.kptss[fi][ki,:2]
                res = -(proj_pt - obs)
                # if res@res>1e3: print('res', res, 'from t', t, 'w', w, 'fi', fi, 'ki', ki)

                d2 = res@res
                alpha = 1
                if d2 > eps2:
                    alpha *= np.sqrt(eps/d2)
                    # print('huber weight', alpha)

                x_,y_,z_ = xform_pt
                Jpi = torch.FloatTensor((1, 0, -x_/z_, 0, 1, -y_/z_)).view(2,3) * torch.FloatTensor((fx/z_, fy/z_)).view(2,1)
                de_dtheta = Jpi @ R.T@crossMatrix(w-t) @ R
                # de_dtheta = Jpi @ R @ R.T@crossMatrix(w-t)
                de_dt = Jpi @ -R.T
                de_dtq = torch.cat((de_dt, de_dtheta), 1)
                de_dw = Jpi @ R.T

                tfi = camTangentId[fi]
                tki = kptTangentId[fi2id[(fi,ki)]]
                de_dtq = de_dtq.cpu().numpy()
                de_dw = de_dw.cpu().numpy()
                for i in range(6): newCoo.extend((newNumResiduals  , tfi+i, de_dtq[0,i]))
                for i in range(3): newCoo.extend((newNumResiduals  , tki+i, de_dw[0,i]))
                for i in range(6): newCoo.extend((newNumResiduals+1, tfi+i, de_dtq[1,i]))
                for i in range(3): newCoo.extend((newNumResiduals+1, tki+i, de_dw[1,i]))
                newNumResiduals += 2
                newRes.append(res)
                newAlpha.append(alpha)
                newAlpha.append(alpha)

                # st[landmarkRow]
            coo = unaryCoo + newCoo

            # Add prior HERE.
            if 1:
                priorCoo = []
                for i in range (NT): priorCoo.extend((newNumResiduals+i,i,.001))
                newNumResiduals += NT
                # newAlpha.extend([1,] * NT)
                newAlpha.extend([.1,] * NT)
                newRes.append(torch.zeros(NT))
                coo = coo + priorCoo

            a = coo[0::3]
            b = coo[1::3]
            c = coo[2::3]
            coo = torch.stack((torch.LongTensor(a),torch.LongTensor(b)),0)
            val = torch.FloatTensor(c)
            newRes = torch.concatenate(newRes,0)
            newResGood = newRes[newRes<newRes.mean()]
            rmse = (newRes**2).sum().sqrt()
            goodRmse = (newResGood**2).sum().sqrt()
            print(' '*80,'-      RMSE:', rmse)
            print(' '*80,'- good-RMSE:', goodRmse)

            M = len(newAlpha)
            Alpha = torch.sparse_coo_tensor(torch.arange(M).view(1,-1).repeat(2,1), torch.FloatTensor(newAlpha)).coalesce()
            J = torch.sparse_coo_tensor(coo,val).coalesce()
            print(J.shape,Alpha.shape,newRes.shape)

            Jtres = J.mT @ Alpha @ newRes
            # Jtres = torch.stack(newRes,0).view(-1)
            print(' - Jtres norm', Jtres.norm())

            dx0 = torch.zeros(J.shape[1])
            # print('J', J.shape)
            # print('Jtres', Jtres.shape, 'newRes', newRes.shape)
            # print('dx0', dx0.shape)
            # print('x', x.shape)
            if 0:
                # NOT WORKING.
                # dx = solve_cg_AtA(J, Jtres, dx0, preconditioned=True, debug=True, T=100)
                dx = solve_cg_AtA_2(J, Alpha, Alpha@Jtres, dx0, preconditioned=True, debug=True, T=100)
            else:
                A = ((J.mT@Alpha)@J).coalesce()
                # WARNING: torch is BROKEN: I have to sort manually...
                if 1:
                    Ainds = A.indices()
                    order = (Ainds[0]*A.size(1)+Ainds[1]).sort().indices
                    Avals = A.values()
                    A = torch.sparse_coo_tensor(Ainds[:,order], Avals[order], size=A.size())
                    dx = solve_cg(A, Jtres, dx0, preconditioned=True, debug=True, T=100)
                else:
                    dx = solve_cg(A, Jtres, dx0, preconditioned=False, debug=True, T=100)
                # print('A:\n',A)
                # print('A:\n',A[0,0])
                # A = A + torch.eye(J.shape[1]).to_sparse() * 10
                # dx = solve_cg(A, Jtres, dx0, preconditioned=True, debug=True, T=100)
            print('got dx:\n', dx)
            dx = dx * .5
            dx_tp = dx[:NC*camTangentSize].view(-1,camTangentSize)
            dx_w = dx[NC*camTangentSize:].view(-1,3)
            for i,dtp in enumerate(dx_tp): x[camStateId[i]:camStateId[i]+7] = box_plus(x[camStateId[i]:camStateId[i]+7], dtp)
            for i,dw in enumerate(dx_w):   x[kptStateId[i]:kptStateId[i]+3] += dw

            if 1:
                cameras = x[:NC*camStateSize]
                cameras = torch.stack([getCamAsMatrix(i) for i in range(NC)], 0).cpu().numpy()
                worldPts = x[NC*camStateSize:].cpu().numpy()
                self.update_viz(
                        cameras,
                        worldPts,
                        worldPtsSeenByEachCamera)

        poses = []
        pts = []
        pass

    def update_viz(self, poses, worldPts, ptsForCams):
        if (not hasattr(self,'viz')) or self.viz is None:
            self.viz = SurfaceRenderer((1024,1024))
            self.viz.init(True)
        viz = self.viz

        worldPts = np.copy(worldPts,'C')
        fcolors = rainbow(np.linspace(0,1,len(poses)))

        while not viz.n_pressed:
            if viz.q_pressed: exit()
            viz.startFrame()
            viz.render()

            # Render everything here.
            for i,pose in enumerate(poses):
                glColor4f(*fcolors[i])
                draw_frustum(pose, self.f,self.wh)

            glColor4f(.5,1,1,1)
            draw_points(worldPts)


            viz.endFrame()
            time.sleep(2e-2)
        viz.startFrame()
        viz.endFrame()

    def find_largest_homog(self):
        ptsa,ptsb,idsa,idsb = self.getVerifiedMatchingPts(0,1)
        H,mask = cv2.findHomography(ptsa,ptsb, cv2.RANSAC, ransacReprojThreshold=1.5)
        mask = mask.reshape(-1)
        assert mask.sum() > 9

        hptsa = ptsa[mask>0]
        hptsb = ptsb[mask>0]
        print(f' - using {mask.sum()}/{len(ptsa)} points for homography')

        h,w = self.frames[0].shape[:2]
        dimg = np.hstack((self.frames[0],self.frames[1]))
        for apt,bpt in zip(ptsa,ptsb):
            cv2.line(dimg,
                        (  int(apt[0]),int(apt[1])),
                        (w+int(bpt[0]),int(bpt[1])), (0,55,155), 1)
        for apt,bpt in zip(hptsa,hptsb):
            cv2.line(dimg,
                        (  int(apt[0]),int(apt[1])),
                        (w+int(bpt[0]),int(bpt[1])), (255,155,0), 1)
        dimg = cv2.resize(dimg, (1536*2, h*1536//w))
        cv2.imshow('dimg',dimg)
        cv2.waitKey(1)

        # Convert mask back to original ids.
        maskIdsa = idsa[mask>0]
        maskIdsb = idsb[mask>0]

        return dict(H=H, maskIdsa=maskIdsa, maskIdsb=maskIdsb)

    def getVerifiedMatchingPts(self, a, b):
        assert a < b
        for entry in self.frameGraph:
            if entry['a'] == a and entry['b'] == b:
                idsa = entry['verifiedIdsa']
                idsb = entry['verifiedIdsb']
                ptsa = self.kptss[a][idsa][:,:2]
                ptsb = self.kptss[b][idsb][:,:2]
                return ptsa,ptsb, idsa,idsb
        raise 'edge not found'


    def matchLinear(self):
        # Match each frame to the next one. If one cannot be matched to the previous, drop it.

        NI = len(self.kptss)
        h,w = self.frames[0].shape[:2]

        graph = []

        for a in range(NI-1):
            b = a + 1
            dimg = np.hstack((self.frames[a],self.frames[b]))

            kptsa,kptsb = self.kptss[a], self.kptss[b]
            desa,desb = self.dess[a], self.dess[b]

            for pt in kptsa: cv2.circle(dimg, (int(  pt[0]),int(pt[1])), 2, (0,255,0), 1)
            for pt in kptsb: cv2.circle(dimg, (int(w+pt[0]),int(pt[1])), 2, (0,255,0), 1)

            ptsa = torch.from_numpy(kptsa[:,:2])
            ptsb = torch.from_numpy(kptsb[:,:2])
            desa = torch.from_numpy(desa).cuda()
            desb = torch.from_numpy(desb).cuda()

            dist = torch.cdist(desa,desb)

            # We have sizes.
            if kptsa.shape[1] > 2:
                sza = torch.from_numpy(kptsa[:,2:3])
                szb = torch.from_numpy(kptsb[:,2:3])
                size_diff = abs(sza - szb.T)
                print('size_diff mask mean', (size_diff>70).float().mean())
                dist += (size_diff > 70).to(dist.device) * 9999

            print(dist.shape)
            d12,assn12 = dist.topk(2,dim=-1,largest=False)
            d21,assn21 = dist.topk(2,dim=-2,largest=False)
            d21,assn21 = d21.T, assn21.T

            idxa = torch.arange(len(ptsa))
            idxb = assn12[:,0].cpu()
            if 1:
                LOWE_RATIO = .9
                good = (d12[:,0] < LOWE_RATIO*d12[:,1]).cpu()
            else:
                # good = (assn12[:,0] == assn21[torch.arange(len(d12),device=assn12.device),0]).cpu()
                # good = (assn12[:,0] == torch.arange(len(d12),device=assn12.device)[assn21[:,0]]).cpu()
                pass

            # good = (d[:,0] < .75*d[:,1]).cpu()
            print(f' - keeping {good.sum()}/{good.numel()} matches')

            # Show all unverified matches
            sptsa = ptsa[idxa[good]].cpu().numpy()
            sptsb = ptsb[idxb[good]].cpu().numpy()

            dimg0 = np.zeros_like(dimg)
            for apt,bpt in zip(sptsa,sptsb):
                cv2.line(dimg0,
                         (  int(apt[0]),int(apt[1])),
                         (w+int(bpt[0]),int(bpt[1])), (0,0,255), 1)

            # Get / Show verified matches
            # E,mask = cv2.findEssentialMat(sptsa,sptsb, self.K, cv2.RANSAC, .999999, threshold=4)
            # Kptsa = project(homogeneous(sptsa) @ self.K.T)
            # Kptsb = project(homogeneous(sptsb) @ self.K.T)
            # E,mask = cv2.findEssentialMat(sptsa,sptsb, self.K, cv2.RANSAC, .999999, threshold=1)
            # E,mask = cv2.findEssentialMat(sptsa,sptsb, self.K, cv2.RANSAC, .999999, threshold=.02 if a == 0 else .3)
            E,mask = cv2.findEssentialMat(sptsa,sptsb, self.K, cv2.RANSAC, .9999999, threshold=.5)
            # E,mask = cv2.findEssentialMat(sptsa,sptsb, self.K, cv2.RANSAC, .9999999, threshold=9.5)
            mask = mask.reshape(-1)
            vptsa = sptsa[mask>0]
            vptsb = sptsb[mask>0]
            print(f' - verified {mask.sum()}/{good.sum()} kept matches')

            verifiedIdsa = idxa[good][mask>0].cpu().numpy()
            verifiedIdsb = idxb[good][mask>0].cpu().numpy()

            for apt,bpt in zip(vptsa,vptsb):
                cv2.line(dimg0,
                         (  int(apt[0]),int(apt[1])),
                         (w+int(bpt[0]),int(bpt[1])), (255,155,0), 1)
            dimg = cv2.addWeighted(dimg,.99, dimg0,.95, 0)

            dimg = cv2.resize(dimg, (1536*2, h*1536//w))
            if (NI) > 3:
                cv2.imshow('dimg',dimg)
            else:
                cv2.imshow('match_{}_{}'.format(a,b),dimg)
            cv2.waitKey(1)

            # graph.append((a,b,E,verifiedIdsa,verifiedIdsb))
            graph.append(dict(a=a,b=b,E=E,verifiedIdsa=verifiedIdsa,verifiedIdsb=verifiedIdsb))
        return graph

    def appendGetIds(self, pts, fid):
        if self.kptss[fid] is None:
            self.kptss[fid] = pts
            return torch.arange(len(pts), dtype=torch.int64)
        else:
            out = torch.zeros(len(pts),dtype=torch.int64)
            new = []
            for i in range(pts.shape[0]):
                j = torch.argwhere(0==abs(self.kptss[fid] - pts[i]).sum(-1))
                if j.numel() > 0:
                    print(j.shape)
                    out[i] = j.item()
                else:
                    out[i] = len(self.kptss[fid]) + len(new)
                    new.append(pts[i])
            self.kptss[fid] = torch.cat((self.kptss[fid],torch.stack(new,0)),0)
        return out

    def matchLinear_loftr(self):
        # Match each frame to the next one. If one cannot be matched to the previous, drop it.

        NI = len(self.frames)
        h,w = self.frames[0].shape[:2]

        graph = []

        for a in range(NI-1):
            b = a + 1

            dimg = np.hstack((self.frames[a],self.frames[b]))

            kptsa,kptsb = run_pair(self.frames[a],self.frames[b])
            # kptsa,kptsb = self.kptss[a], self.kptss[b]
            # desa,desb = self.dess[a], self.dess[b]

            for pt in kptsa: cv2.circle(dimg, (int(  pt[0]),int(pt[1])), 2, (0,255,0), 1)
            for pt in kptsb: cv2.circle(dimg, (int(w+pt[0]),int(pt[1])), 2, (0,255,0), 1)

            sptsa = kptsa
            sptsb = kptsb

            dimg0 = np.zeros_like(dimg)
            for apt,bpt in zip(sptsa,sptsb):
                cv2.line(dimg0,
                         (  int(apt[0]),int(apt[1])),
                         (w+int(bpt[0]),int(bpt[1])), (0,0,255), 1)

            # Get / Show verified matches
            # E,mask = cv2.findEssentialMat(sptsa,sptsb, self.K, cv2.RANSAC, .999999, threshold=4)
            # Kptsa = project(homogeneous(sptsa) @ self.K.T)
            # Kptsb = project(homogeneous(sptsb) @ self.K.T)
            # E,mask = cv2.findEssentialMat(sptsa,sptsb, self.K, cv2.RANSAC, .999999, threshold=1)
            # E,mask = cv2.findEssentialMat(sptsa,sptsb, self.K, cv2.RANSAC, .999999, threshold=.02 if a == 0 else .3)
            # E,mask = cv2.findEssentialMat(sptsa,sptsb, self.K, cv2.RANSAC, .9999999, threshold=.5)
            E,mask = cv2.findEssentialMat(sptsa,sptsb, self.K, cv2.RANSAC, .9999999, threshold=9.5)
            mask = mask.reshape(-1)
            vptsa = sptsa[mask>0]
            vptsb = sptsb[mask>0]
            print(f' - verified {mask.sum()}/{sptsa.shape[0]} kept matches')


            verifiedIds = torch.arange(mask.shape[0])[mask>0].cpu().numpy()
            idsInA = self.appendGetIds(torch.from_numpy(kptsa),a)
            idsInB = self.appendGetIds(torch.from_numpy(kptsb),b)
            verifiedIdsa = idsInA[mask>0].cpu().numpy()
            verifiedIdsb = idsInB[mask>0].cpu().numpy()

            for apt,bpt in zip(vptsa,vptsb):
                cv2.line(dimg0,
                         (  int(apt[0]),int(apt[1])),
                         (w+int(bpt[0]),int(bpt[1])), (255,155,0), 1)
            dimg = cv2.addWeighted(dimg,.99, dimg0,.95, 0)

            dimg = cv2.resize(dimg, (1536*2, h*1536//w))
            if (NI) > 3:
                cv2.imshow('dimg',dimg)
            else:
                cv2.imshow('match_{}_{}'.format(a,b),dimg)
            cv2.waitKey(0)

            # graph.append((a,b,E,verifiedIdsa,verifiedIdsb))
            graph.append(dict(a=a,b=b,E=E,verifiedIdsa=verifiedIdsa,verifiedIdsb=verifiedIdsb))
        return graph





with torch.no_grad():
    # run = Runner('/data/chromeDownloads/20230628_164313.mp4')
    # run = Runner('/data/chromeDownloads/20230628_164313.mp4')
    # run = Runner('/data/chromeDownloads/apt')
    run = Runner('/data/chromeDownloads/apt2')
    # run = Runner('/data/chromeDownloads/20230630_150047.mp4')

