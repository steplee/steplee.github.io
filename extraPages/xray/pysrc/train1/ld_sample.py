import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from torch.utils.data import Dataset, DataLoader

from ..data.posePriorDataset import PosePriorDataset, Map_PP_to_Coco_v1
from ..loss_dict import LossDict
import os, sys, cv2

from .models import *
from ..est2d.run import FasterRcnnModel, get_resnet50, run_model_and_viz

from matplotlib.cm import gist_rainbow as hsv

def crop_image(img):
    h,w = img.shape[:2]
    e = (min(h,w) // 2) * 2
    img = img[h//2-e//2:h//2+e//2, w//2-e//2:w//2+e//2]
    img = cv2.resize(img, (512,512))
    return img


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

        self.meta.setdefault('sigma1', 1)
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
            z[:,1] = 511 - z[:,1]
            z = z.view(1,z.size(0),z.size(1)).repeat(B,1,1)
            print('z shape', z.shape)


            # Interpolate sigma0 to sigma1 exponentially.
            T = 60
            base = (self.sigma0/self.sigma1) ** (1./T)
            all_ts = [self.sigma1 * (base ** t) for t in range(T)]
            all_ts = torch.cuda.FloatTensor(all_ts)
            print(f' - going from {all_ts[0]:.3f} to {all_ts[-1]:.3f}')

            all_xs = []
            all_xs.append(x.clone().cpu())

            for i in range(T):
                # score = self.model3d(
                # ts = torch.rand(B, 1, device=D)
                # sigs = self.t_to_sigma(ts)
                # rs = torch.randn(B, S, device=D) * sigs

                ts = all_ts[i].view(1).repeat(B)
                s = self.model3d(x,ts,z)

                new_randomness = torch.randn_like(s) * ts.view(B,1) * .2

                # Add!
                x += s*.7 + new_randomness
                all_xs.append(x.clone().cpu())

            # [T+1,B,L,3] tensor
            all_xs = torch.stack(all_xs,0).numpy().reshape(T+1, B, -1, 3)
            all_xs = np.copy(all_xs,'C')
            L = all_xs.shape[-2]
            print(' - all_xs', all_xs.shape)

            inds0 = self.ppDset.inds
            inds = np.concatenate([inds0+L*i for i in range(B)])

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
