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

def get_intrin(vfov, wh):
    v2 = np.tan(np.deg2rad(vfov)*.5)
    f = np.array((wh[1]*.5/v2, wh[1]*.5/v2))
    c = np.array((wh[0]/2,wh[1]/2))
    K = np.array((
        f[0],0,c[0],
        0,f[1],c[1],
        0,0,1)).astype(np.float32).reshape(3,3)
    return f,c,K # lol.

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
    if isinstance(rvec, np.ndarray): rvec = torch.from_numpy(rvec).float()
    if isinstance(tvec, np.ndarray): tvec = torch.from_numpy(tvec).float()
    print('WARNING: make sure this is correct and compatiable with cv2')
    if invert:
        qc = q_exp(-rvec)
        # print(q_to_rot(qc))
        # print(np.linalg.det(q_to_rot(qc)))
        return torch.cat((q_to_rot(qc) @ -tvec, qc))
    else:
        return torch.cat((tvec, q_exp(rvec)))

def homogeneous(a):
    if isinstance(a,np.ndarray):
        return np.concatenate((a,np.ones_like(a[:,-1:])), -1)
    else:
        return torch.cat((a,torch.ones_like(a[:,-1:])), -1)
def project(a):
    return a[...,:-1] / a[...,-1:]

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

class Runner:
    def __init__(self, file, isSequential=True):

        self.file = file
        self.frames = get_frames(file)

        self.f,self.c,self.K = get_intrin(31.5, self.frames[0].shape[:2][::-1])
        self.wh=self.c*2
        self.N = len(self.frames)

        # self.kptss = [None for _ in range(len(self.frames))]
        # self.frameGraph = self.matchLinear_loftr()
        # self.kptss = [k.cpu().numpy().astype(np.float32) for k in self.kptss]

        # first_H_info = self.find_largest_homog() # This'll dictate the "ground" plane.
        # first_H_info = None
        # poseGraph = self.construct_graph(self.frameGraph, first_H_info)

        self.connectionHints = []
        if isSequential:
            self.connectionHints.extend([(i,j) for i,j in zip(range(self.N-1), range(1,self.N))][::-1])

        self.matchBasedOnHints()

    def matchBasedOnHints(self):
        while self.connectionHints:
            ai,bi = self.connectionHints.pop()

            kptsa,kptsb = run_pair(self.frames[ai],self.frames[bi])


def test_data_struct():
    # apts = torch.randint(0,256, size=(100,2))
    # bpts = torch.randint(0,256, size=(100,2))
    # am = torch.randint(0,256, size=(100))
    # bm = torch.randint(0,256, size=(100))
    apts = torch.LongTensor((
        0, 5,5,
        0, 6,5,
        0, 7,5,
        0, 8,8,)).view(-1,3)
    bpts = torch.LongTensor((
        1, 4,4,
        1, 8,8,)).view(-1,3)
    cpts = torch.LongTensor((
        2, 9,9,
        2, 8,8,)).view(-1,3)
    allpts = torch.cat((apts,bpts,cpts),0)
    # Stores global id as value.
    pts_st = torch.sparse_coo_tensor(allpts.t(), torch.arange(allpts.size(0),dtype=torch.long)).coalesce()
    print('pts_st:\n',pts_st)

    allmatches = torch.LongTensor((
        0, 5,5, 0,
        1, 4,4, 1,
        0, 8,8, 0,
        1, 8,8, 1,

        1, 8,8, 0,
        2, 9,9, 1,
        )).view(-1,4)
    img_matches_st = torch.sparse_coo_tensor(allmatches.t(), torch.ones(allmatches.size(0),dtype=torch.long)).coalesce()
    print('img_matches_st:\n',img_matches_st)

    # state_matches_a = img_matches_st.indices().t()[ ::2][...,0:3]
    # state_matches_b = img_matches_st.indices().t()[1::2][...,0:3]
    inds = img_matches_st.indices().t()

    '''
    # Even-odd decomposition here
    state_matches_e = inds[inds[:,3]==0][...,0:3]
    state_matches_o = inds[inds[:,3]==1][...,0:3]

    # We can create the jacobian blocks for (cam_a <-> pts_a) and likewise for _b
    # We'll need to lookup the global indices of the observed img pts
    state_matches_e_st = torch.sparse_coo_tensor(state_matches_e.t(), torch.ones(state_matches_e.size(0),dtype=torch.long), size=pts_st.size()).coalesce()
    state_matches_o_st = torch.sparse_coo_tensor(state_matches_o.t(), torch.ones(state_matches_o.size(0),dtype=torch.long), size=pts_st.size()).coalesce()
    print('state_matches_a_st',state_matches_e_st.shape)
    print('state_matches_b_st',state_matches_o_st.shape)
    print('state_matches_a_st',state_matches_e_st)
    print('state_matches_b_st',state_matches_o_st)
    matches_e_gid = (pts_st * state_matches_e_st).coalesce().values()
    matches_o_gid = (pts_st * state_matches_o_st).coalesce().values()
    print('global ids for the matched e pts:\n', matches_e_gid)
    print('global ids for the matched o pts:\n', matches_o_gid)
    print('camera ids for the matched e pts:\n', state_matches_e_st.indices()[0])
    print('camera ids for the matched o pts:\n', state_matches_o_st.indices()[0])
    '''
    # state_matches_st = torch.sparse_coo_tensor(a_b_matches.t(), torch.ones(a_b_matches.size(0)))

    state_matches_all = inds[...,0:3]
    state_matches_all_st = torch.sparse_coo_tensor(state_matches_all.t(), torch.ones(state_matches_all.size(0),dtype=torch.long), size=pts_st.size()).coalesce()
    print('state_matches_all_st:\n',state_matches_all_st)
    matches_all_gid = (pts_st * state_matches_all_st).coalesce()
    matches_all_gid_vals = matches_all_gid.values()
    matches_all_gid_cams = matches_all_gid.indices()[0] # etc...
    print('all matches cameras   :', matches_all_gid_cams)
    print('all matches global ids:', matches_all_gid_vals)

    '''
    Okay so we store the `allpts` and `allmatches` __dense__ tensors like above. We can dynamically add to them.

    Then we create the img_matches_st sparse tensor when ready for graph optimization.
    To construct sparse jacobian blocks, we further need the `state_matches_all_st`.
    This nearly gives the structure of the jacobian, we just need to account for camera/point offset and also camera/point sizes being >1.
    '''

    exit()
test_data_struct()

with torch.no_grad():
    # run = Runner('/data/chromeDownloads/20230628_164313.mp4')
    # run = Runner('/data/chromeDownloads/20230628_164313.mp4')
    # run = Runner('/data/chromeDownloads/apt')
    run = Runner('/data/chromeDownloads/apt2')
    # run = Runner('/data/chromeDownloads/20230630_150047.mp4')
