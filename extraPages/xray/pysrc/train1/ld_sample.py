import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from torch.utils.data import Dataset, DataLoader

from ..data.posePriorDataset import PosePriorDataset, Map_PP_to_Coco_v1
from ..loss_dict import LossDict
import os, sys, cv2

from .models import *
from ..est2d.run import FasterRcnnModel, get_resnet50, run_model_and_viz

from matplotlib.cm import gist_rainbow as hsv
torch.set_printoptions(linewidth=200)

def crop_image(img):
    h,w = img.shape[:2]
    e = (min(h,w) // 2) * 2
    img = img[h//2-e//2:h//2+e//2, w//2-e//2:w//2+e//2]
    img = cv2.resize(img, (512,512))
    return img

def choose_median_skeleton(x):
    B,S = x.size()
    d = torch.cdist(x,x)

    # dd = d.sum(0)
    dd = d.pow(2).sum(0)

    idx = dd.min(0).indices
    # Return also with batch-dim
    return x[idx:idx+1]

def q_to_R(q):
    # Assumes unit-norm!
    q1,q2,q3,q0 = q
    R = torch.FloatTensor((
        q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2-q0*q3), 2*(q0*q2+q1*q3),
        2*(q1*q2+q0*q3), (q0*q0-q1*q1+q2*q2-q3*q3), 2*(q2*q3-q0*q1),
        2*(q1*q3-q0*q2), 2*(q0*q1+q2*q3), q0*q0-q1*q1-q2*q2+q3*q3)).view(3,3)
    return R
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
    # print(q_to_matrix1(q).T.shape,pts.shape, (pts-t.view(1,3)).T.shape)
    # tpts = (q_to_matrix1(q).T @ (pts - t.view(1,3)).T).T
    tpts = (pts-t.view(-1,3)) @ q_to_matrix1(q).mT
    # print( q_to_matrix1(q).mT)
    return (tpts[...,:2] / tpts[...,2:]) * fxy + wh*.5

def get_functorch_jac_2():
    from functorch import jacfwd, vmap
    from functorch.compile import aot_function, memory_efficient_fusion, ts_compile

    # Each camera projects the SAME number of points
    m2 = vmap(mapProject2, (0,0,0,0))
    d_m2 = vmap(jacfwd(mapProject2, (0,1)), (0,0,0,0))

    return m2, d_m2

# This is not so good
def recover_camera(wh, fxy, obspts, worldpts,
                   initialEye=torch.FloatTensor((0,1,-1)),
                   initialQ=torch.FloatTensor((1,0,0,0)),
            ):

    B,N,three = worldpts.size()
    x = torch.cat((initialQ,initialEye), -1).view(1,7).repeat(B,1)
    wh = wh.view(1,2).repeat(B,1)
    fxy = fxy.view(1,2).repeat(B,1)

    Nstate = 7
    Nobs = N * 2

    prior0 = torch.eye(7).unsqueeze_(0)
    prior0[:,:4,:4] *= 100
    prior0[:,4:,4:] *= .1
    prior = prior0.clone()

    F, dF = get_functorch_jac_2()

    print('obs\n',obspts)

    for i in range(8):
        pred = F(x, worldpts, fxy, wh)
        Js = dF(x, worldpts, fxy, wh)

        # print('pred\n',pred)
        res = pred - obspts # [B,N,2]
        rmse = res.pow(2).sum(2).mean(1).sqrt() # [B]
        print(f' - step {i} rmse {rmse}')

        # print(pred.shape,[JJ.shape for JJ in Js])
        J = Js[0].sum(1) # sum out 'N', the observation dim. to get a [B,2,7] tensor.

        grad = (J.mT @ res.mT).sum(2) # [B,2,7] x [B,N,2] -> [B,7]

        JtJ = J.mT @ J # [B,7,7]
        # print('JtJ:\n',JtJ)
        # print('grad:',grad)
        Hess = JtJ + prior
        P = Hess.inverse()
        # print('P:\n',P)

        if (P.diagonal(dim1=1,dim2=2) <= 0).any():
            print(P.diagonal(dim1=1,dim2=2))
            print(' - WARNING: non-pos-definite covariance.')
            prior = prior*2
            # continue
            exit()

        # print(P.shape,grad.shape)
        x = x - (P @ grad.unsqueeze(-1))[...,0]

        x[...,:4] = nn.functional.normalize(x[...,:4], dim=-1)
        # x[...,:4] = torch.FloatTensor((1,0,0,0)).view(1,4)
        print('q',x[0,:4], 't',x[0,4:])

torch.manual_seed(0)
wh = torch.FloatTensor((512,512.))
uv = np.tan(np.deg2rad(50)/2)*2
fxy = wh/uv
worldpts = torch.randn(1,10,3) + torch.FloatTensor((0,1,8)).view(1,1,3)

# def mapProject2(pose, pts, fxy, wh):
# obspts = (worldpts * 1)[:,:,:2] * fxy + wh*.5
pose0 = torch.FloatTensor((1,0,0,0, 0,2.1,-1.02)).view(7)
obspts = mapProject2(pose0, worldpts, fxy, wh)

recover_camera(wh, fxy, obspts, worldpts)
# exit()


####################################################
# Rendering Code.
####################################################


from .renderExamples import *

class LD_Renderer(ExampleRendererBase):
    def __init__(self, h, w):
        super().__init__(h,w)

        self.inds,self.x = None, None
        self.addSign = 1

        self.animationTime = -1


    def render(self):
        glViewport(0, 0, *self.wh)
        glMatrixMode(GL_PROJECTION)
        n = .001
        v = .5
        u = (v*self.wh[0]) / self.wh[1]
        glLoadIdentity()
        glFrustum(-u*n,u*n,-v*n,v*n,n,100)

        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(self.view.T.reshape(-1))

        glClearColor(0,0,0,1.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.animationTime >= 0:
            self.renderSkeletonAnimated()

    def setData(self, inds, x):
        self.inds = inds.astype(np.uint32)
        self.animationTime = 0

        print(x.shape)
        T,B,L,_ = x.shape
        color = hsv(np.linspace(0,1,B)).astype(np.float32)
        print(np.linspace(0,1,B))
        print(color.shape)
        color[...,3] = 1
        color = color.reshape(1,B,1,4) + np.zeros((T,B,L,4),dtype=np.float32)
        print(x.shape,color.shape, x.dtype,color.dtype)
        x = np.concatenate((x,color), -1)

        self.animEst = x


    def renderSkeletonAnimated(self):
        I = self.animationTime
        assert I >= 0

        x = self.animEst[I]

        for xx in (x,):
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            glVertexPointer(3,GL_FLOAT,28,xx)
            glColorPointer(4,GL_FLOAT, 28, ctypes.c_void_p(xx.ctypes.data+12))
            glDrawElements(GL_LINES,self.inds.size,GL_UNSIGNED_INT,self.inds)
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)

    def keyboard(self, key, x, y):
        super().keyboard(key,x,y)
        key = (key).decode()
        if key == '[': self.animationTime = max(self.animationTime - 1, 0)
        if key == ']': self.animationTime = min(self.animationTime + 1, self.animEst.shape[0]-1)
        if key == 'n': self.n_pressed = True
        if key == 'w': self.accTrans[2] = -1
        if key == 's': self.accTrans[2] = 1
        if key == 'a': self.accTrans[0] = -1
        if key == 'd': self.accTrans[0] = 1
        if key == 'f': self.addSign *= -1
        self.lastKey = key













####################################################
# Actual Sampling Code.
####################################################


class Sampler:
    def __init__(self,
                 meta,
                 model3d,
                 model2d,
                 ppDset,
                 basePoseIdx,
                 ):
        for k,v in locals().items(): setattr(self,k,v)

        basePose = self.ppDset[basePoseIdx]
        print(basePose)
        self.basePose = basePose.cuda()

        self.meta.setdefault('sigma1', 1.2)
        self.meta.setdefault('sigma0', .02)
        self.sigma1, self.sigma0 = self.meta['sigma1'], self.meta['sigma0']

    def sample(self, img, B=64):
        with torch.no_grad():
            S = self.basePose.size(0)
            x = self.basePose.view(1,S) + torch.randn(B,S,device=self.basePose.device)
            print(x)

            # Extract coco skeleton from the image.
            assert img.shape[0] == 512
            assert img.shape[1] == 512
            assert img.shape[2] == 3

            ximg = torch.from_numpy(img).cuda()
            out = self.model2d(ximg)[0]
            keypoints = out['keypoints'][0]
            print('kpts shape', out['keypoints'].shape)
            print('kpts shape', keypoints.shape)
            # print(keypoints)
            z = keypoints[:, :2].reshape(-1,2).cuda()
            # z = (z - 256) * 1.4 + 256
            # z[:,1] = 511 - z[:,1]
            z = z.view(1,z.size(0),z.size(1)).repeat(B,1,1)
            print('z shape', z.shape)


            # Interpolate sigma0 to sigma1 exponentially.
            T = 60
            base = (self.sigma0/self.sigma1) ** (1./T)
            all_sigs = [self.sigma1 * (base ** t) for t in range(T)]
            all_sigs = torch.cuda.FloatTensor(all_sigs)
            print(f' - going from sigma {all_sigs[0]:.3f} to {all_sigs[-1]:.3f}')

            all_xs = []

            # xx = choose_median_skeleton(x)
            xx = x
            all_xs.append(xx.clone().cpu())

            for i in range(T):
                # score = self.model3d(
                # ts = torch.rand(B, 1, device=D)
                # sigs = self.t_to_sigma(ts)
                # rs = torch.randn(B, S, device=D) * sigs

                ts  = torch.cuda.FloatTensor([1. - i/T]).view(1).repeat(B)
                sig = all_sigs[i].view(1).repeat(B)
                s = self.model3d(x,ts,z)

                new_randomness = torch.randn_like(s) * sig.view(B,1) * .2

                # Add!
                x += s*.5 + new_randomness

                # xx = choose_median_skeleton(x)
                xx = x
                all_xs.append(xx.clone().cpu())

            BB = all_xs[0].size(0)

            # [T+1,BB,L,3] tensor
            all_xs = torch.stack(all_xs,0).numpy().reshape(T+1, BB, -1, 3)
            all_xs = np.copy(all_xs,'C')
            L = all_xs.shape[-2]
            print(' - all_xs', all_xs.shape)

            inds0 = self.ppDset.inds
            inds = np.concatenate([inds0+L*i for i in range(BB)])

            renderer = LD_Renderer(1024,1364)
            renderer.init(True)
            renderer.setData(inds, all_xs)
            while True:
                renderer.startFrame()
                renderer.render()
                renderer.endFrame()
                if renderer.q_pressed: break


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dsetFile', default='/data/human/posePrior/procStride10/data.npz')
    parser.add_argument('--basePoseIdx', default=0, type=int)
    parser.add_argument('--load', default='/data/human/saves/firstModel1.30704.pt')
    parser.add_argument('--image', default='./data/me3.jpg')
    args = parser.parse_args()

    meta = {}

    ppDset = PosePriorDataset(args.dsetFile, masterScale=.03)

    model3d,model3d_meta = get_model(dict(load=args.load))
    model3d = model3d.cuda().eval()

    model2d = FasterRcnnModel(get_resnet50())
    img = cv2.imread(args.image)[...,[2,1,0]]
    img = crop_image(img)
    run_model_and_viz(model2d, img)

    sampler = Sampler(meta, model3d, model2d, ppDset, args.basePoseIdx)
    sampler.sample(img)
