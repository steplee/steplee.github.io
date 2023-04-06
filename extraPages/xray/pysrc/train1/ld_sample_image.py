import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from torch.utils.data import Dataset, DataLoader

from ..data.posePriorDataset import PosePriorDataset, Map_PP_to_Coco_v1
from ..loss_dict import LossDict
import os, sys, cv2

from .models import *
from ..est2d.run import FasterRcnnModel, get_resnet50, run_model_and_viz, get_coco_skeleton
from ..solver.pnp import recover_camera, q_to_matrix1

from matplotlib.cm import gist_rainbow as hsv
torch.set_printoptions(linewidth=200)

torch.manual_seed(0)

# TODO:
# FIXME: Show projected *ESTIMATED* skeleton on far plane.


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
    # q1,q2,q3,q0 = q
    q0,q1,q2,q3 = q
    R = torch.FloatTensor((
        q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2-q0*q3), 2*(q0*q2+q1*q3),
        2*(q1*q2+q0*q3), (q0*q0-q1*q1+q2*q2-q3*q3), 2*(q2*q3-q0*q1),
        2*(q1*q3-q0*q2), 2*(q0*q1+q2*q3), q0*q0-q1*q1-q2*q2+q3*q3)).view(3,3)
    return R

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

        self.viewOne, self.viewIdx = True, 0
        self.imgTex = 0


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

        glEnable(GL_DEPTH_TEST)

        if self.animationTime >= 0:
            self.renderSkeletonAnimated()

    def setData(self, inds, x,
                camPoses=None, fxy=None, wh=None, zs=None, img=None,
                cocoSkeletons3d=None, cocoInds=None,
                ):
        self.inds = inds.astype(np.uint32)
        self.animationTime = 0

        # print(x.shape)
        T,B,L,_ = x.shape
        color = hsv(np.linspace(0,1,B)).astype(np.float32)
        # print(color.shape)
        color[...,3] = 1
        color = color.reshape(1,B,1,4) + np.zeros((T,B,L,4),dtype=np.float32)
        # print(x.shape,color.shape, x.dtype,color.dtype)
        x = np.concatenate((x,color), -1).astype(np.float32)

        self.animEst = x

        self.cocoSkeletons3d,self.cocoInds = cocoSkeletons3d, cocoInds.astype(np.uint16)

        # self.camPoses = camPoses
        self.zs = zs
        # print('ZS',zs)
        self.fxy = fxy
        self.img = img
        self.camMats = []
        self.obsPtsLines = []
        if camPoses is not None:
            # u,v = 1 * fxy / wh
            u,v = wh / fxy
            uu,vv = .5*u, .5*v
            for i in range(len(camPoses)):
                # print(camPoses[i])
                q = camPoses[i,0:4]
                eye = camPoses[i,4:7]
                # R = q_to_R(q)
                R = q_to_matrix1(q)

                CIV = np.eye(4,dtype=np.float32)
                CP = np.eye(4,dtype=np.float32)
                CIV[:3,:3] = R
                CIV[:3,3 ] = eye
                n,f = .1,1
                CP[:] = np.array((
                    1/uu, 0,0,0,
                    # 0, -1/vv, 0,0,
                    0, 1/vv, 0,0, # NOTE: Not flipped
                    0,0, (f+n)/(f-n), -2*f*n/(f-n),
                    0,0, 1,0),dtype=np.float32).reshape(4,4)
                model = CIV @ np.linalg.inv(CP)

                # pts0 = zs.reshape(-1,2) / wh - .5
                # pts0 = (zs.reshape(-1,2) * .5) * wh / fxy
                pts0 = (zs.reshape(-1,2))
                # print('PTS0',pts0)

                ptsFar = np.ones((pts0.shape[0], 4), dtype=np.float32)
                ptsFar[:,:2] = pts0
                ptsFar[:,2] *= 1+1
                ptsFar = (ptsFar @ (model.T))

                ptsNear = np.ones((pts0.shape[0], 4), dtype=np.float32)
                ptsNear[:,:2] = pts0
                ptsNear[:,2] *= 1
                ptsNear = (ptsNear @ (model.T))

                self.camMats.append(model)
                self.obsPtsLines.append(np.stack((ptsNear,ptsFar),-2))
            self.camMats = np.stack(self.camMats)
            self.obsPtsLines = np.stack(self.obsPtsLines)
            # print('OBSPTSLINES',self.obsPtsLines.shape)
        else:
            self.camMats = self.obsPtsLines = None

        # print('IMG', img.shape)
        if img is not None:
            if self.imgTex > 0: glDeleteTextures([self.imgTex])
            self.imgTex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.imgTex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.shape[1], img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
            glBindTexture(GL_TEXTURE_2D, 0)


    def renderCam(self, viewIdx):

        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glEnable(GL_BLEND)

        # Lines connecting points.
        glColor4f(1,0,1,.3)
        # pts1 = np.hstack((pts, ptsFar)).reshape(-1,4)
        pts1 = self.obsPtsLines[viewIdx]
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(4,GL_FLOAT,0,pts1)
        glDrawArrays(GL_LINES,0,len(pts1)*2)
        glDisableClientState(GL_VERTEX_ARRAY)

        # Frustum.
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        mv = self.view @ self.camMats[viewIdx]
        glLoadMatrixf(mv.T.reshape(-1))
        glColor4f(1,1,1,1)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3,GL_FLOAT,0,frustum_pts)
        glDrawElements(GL_LINES,frustum_inds.size,GL_UNSIGNED_SHORT,frustum_inds)
        glDisableClientState(GL_VERTEX_ARRAY)
        glPopMatrix()

        # Image.
        if self.imgTex > 0:
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            mv = self.view @ self.camMats[viewIdx]
            glLoadMatrixf(mv.T.reshape(-1))
            glColor4f(1,1,1,1)
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.imgTex)
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            glVertexPointer(3,GL_FLOAT,0,frustum_pts[4:])
            glTexCoordPointer(2,GL_FLOAT,0,frustum_uvs)
            glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_SHORT,frustum_face_inds)
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            glBindTexture(GL_TEXTURE_2D, 0)
            glDisable(GL_TEXTURE_2D)
            glPopMatrix()



    def renderSkeletonAnimated(self):
        I = self.animationTime
        assert I >= 0

        x = self.animEst[I]

        glEnable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

        if self.viewOne:
            # glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE);
            viewIdx = self.viewIdx % x.shape[0]
            x = np.copy(x)
            x[...,6] = min(max(.01, .05 * (32 / x.shape[0])), .1)
            x[viewIdx,:,6] = 1.

            if self.cocoSkeletons3d is not None:
                glDisable(GL_DEPTH_TEST)
                glLineWidth(4)
                glColor4f(.9,.3,0,.5)
                cx = self.cocoSkeletons3d[I,viewIdx]
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(3,GL_FLOAT,0,cx)
                glDrawElements(GL_LINES,self.cocoInds.size,GL_UNSIGNED_SHORT,self.cocoInds)
                glDisableClientState(GL_VERTEX_ARRAY)
                glEnable(GL_DEPTH_TEST)
                glLineWidth(1)

            if self.camMats is not None:
                self.renderCam(viewIdx)


        else:
            # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA)

        glDisable(GL_DEPTH_TEST)

        glColor4f(1,1,1,1.)
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

        if key == 'm': self.viewOne = not self.viewOne
        if key == 'n': self.viewIdx += 1
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
        # print(basePose)
        self.basePose = basePose.cuda()

        self.meta.setdefault('sigma1', 1.2)
        self.meta.setdefault('sigma0', .02)
        self.sigma1, self.sigma0 = self.meta['sigma1'], self.meta['sigma0']

        coco_inds, cocoJoints = get_coco_skeleton()
        self.coco_inds = np.array(coco_inds).reshape(-1)
        self.to_coco = Map_PP_to_Coco_v1(cocoJoints,self.ppDset.joints)

    def sample(self, img, B=64):
        with torch.no_grad():
            S = self.basePose.size(0)
            x = self.basePose.view(1,S) + torch.randn(B,S,device=self.basePose.device)
            # print(x)

            camFxy = torch.FloatTensor((600,600)).cuda()
            camWh = torch.FloatTensor((512,512)).cuda()

            # Extract coco skeleton from the image.
            assert img.shape[0] == 512
            assert img.shape[1] == 512
            assert img.shape[2] == 3

            # ximg = torch.from_numpy(img).cuda()
            # out = self.model2d(ximg)[0]
            out, vimg = run_model_and_viz(self.model2d, img, show=False)
            img = vimg

            keypoints = out['keypoints'][0]
            print('kpts shape', out['keypoints'].shape)
            print('kpts shape', keypoints.shape)
            # print(keypoints)
            estSkelPts = keypoints[:, :2].reshape(-1,2).cuda()
            # z = (z - 256) * 1.4 + 256
            # z[:,1] = 511 - z[:,1]
            estSkelPts = estSkelPts.view(1,estSkelPts.size(0),estSkelPts.size(1)).repeat(B,1,1)
            # print('estSkelPts shape', estSkelPts.shape)
            # Zs are in range [-1,1] in both x/y axes
            z = (estSkelPts - camWh*.5) / (camWh * .5)
            # print('z',z)


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

            NOISE_WEIGHT = .3
            MODEL_WEIGHT = .8

            print(' - Doing Reverse Diffusion Process...')

            for i in range(T):
                # score = self.model3d(
                # ts = torch.rand(B, 1, device=D)
                # sigs = self.t_to_sigma(ts)
                # rs = torch.randn(B, S, device=D) * sigs

                ts  = torch.cuda.FloatTensor([1. - i/T]).view(1).repeat(B)
                sig = all_sigs[i].view(1).repeat(B)
                s = self.model3d(x,ts,z)

                new_randomness = torch.randn_like(s) * sig.view(B,1) * NOISE_WEIGHT

                # Add!
                x += s * MODEL_WEIGHT + new_randomness

                # xx = choose_median_skeleton(x)
                xx = x
                all_xs.append(xx.clone().cpu())

            print(' -                                ... Done')

            BB = all_xs[0].size(0)

            # [T+1,BB,L,3] tensor
            all_xs = torch.stack(all_xs,0).reshape(T+1, BB, -1, 3)
            L = all_xs.shape[-2]
            all_xs_numpy = all_xs.cpu().numpy()
            # print(' - all_xs', all_xs.shape)

            inds0 = self.ppDset.inds
            inds = np.concatenate([inds0+L*i for i in range(BB)])


            if 1:
                cocoSkeletons3d = self.to_coco(all_xs.reshape(-1, L*3)).reshape(T+1, BB, -1, 3).cpu().numpy()

            if 1:
                obspts = estSkelPts.cpu()
                # print(obspts)
                # img = None

                camPoses = torch.FloatTensor((1,0,0,0, 0,0,-3)).view(1,7).repeat(BB,1)

                # q0=torch.FloatTensor((1,0,0,0))
                q0=torch.FloatTensor((0,1,0,0))
                t0=torch.FloatTensor((0,1,2))

                # Use last time step only for now
                print(' - Recovering cameras...')
                camPoses = recover_camera(camWh.cpu(), camFxy.cpu(), obspts, torch.from_numpy(cocoSkeletons3d)[-1], initialQ=q0,initialEye=t0)
                print(' -                   ... Done')

            else:
                camPoses,camFxy,camWh,obspts,img = (None,)*5

            renderer = LD_Renderer(1024,1364)
            renderer.init(True)
            print(' - Setting renderer data...')
            renderer.setData(inds, all_xs_numpy,
                             cocoSkeletons3d=cocoSkeletons3d, cocoInds=self.coco_inds,
                             camPoses = camPoses, fxy=camFxy.cpu(), wh=camWh.cpu(), zs=z.cpu(), img=img
                             )
            print(' - Setting renderer data... Done')


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
    parser.add_argument('-n', default=64, type=int)
    args = parser.parse_args()

    meta = {}

    ppDset = PosePriorDataset(args.dsetFile, masterScale=.03)

    model3d,model3d_meta = get_model(dict(load=args.load))
    model3d = model3d.cuda().eval()

    model2d = FasterRcnnModel(get_resnet50().cuda().eval())
    img = cv2.imread(args.image)[...,[2,1,0]]
    img = crop_image(img)
    # run_model_and_viz(model2d, img)

    sampler = Sampler(meta, model3d, model2d, ppDset, args.basePoseIdx)
    sampler.sample(img, B=args.n)
