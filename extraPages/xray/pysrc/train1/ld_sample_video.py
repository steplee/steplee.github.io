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
        self.imgTexs = []
        self.opacity = 1


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
                camPosess=None, fxy=None, wh=None, zs=None, imgs=None,
                cocoSkeletons3d=None, cocoInds=None,
                ):
        self.inds = inds.astype(np.uint32)
        self.animationTime = 0
        self.imgWH = imgs[0].shape[0:2][::-1]


        # print(x.shape)
        N,B,L,_ = x.shape
        color = hsv(np.linspace(0,1,B)).astype(np.float32)
        # print(color.shape)
        color[...,3] = 1
        color = color.reshape(1,B,1,4) + np.zeros((N,B,L,4),dtype=np.float32)
        x = np.concatenate((x,color), -1).astype(np.float32)


        self.animEst = x

        self.cocoSkeletons3d,self.cocoInds = cocoSkeletons3d, cocoInds.astype(np.uint16)

        self.zs = zs
        self.fxy = fxy
        self.imgs = imgs
        if camPosess is not None:
            # u,v = 1 * fxy / wh
            u,v = wh / fxy
            uu,vv = .5*u, .5*v
            self.camMats = []
            self.obsPtsLines = []
            for ni in range(len(camPosess)):
                camPoses = camPosess[ni]
                obsPtsLines, camMats = [], []
                for bi in range(len(camPoses)):
                    q = camPoses[bi,0:4]
                    eye = camPoses[bi,4:7]
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

                    camMats.append(model)
                    obsPtsLines.append(np.stack((ptsNear,ptsFar),-2))
                self.camMats.append(np.stack(camMats))
                self.obsPtsLines.append(np.stack(obsPtsLines))
            self.camMats = (np.stack(self.camMats))
            self.obsPtsLines = (np.stack(self.obsPtsLines))
            # print('OBSPTSLINES',self.obsPtsLines.shape)
        else:
            self.camMats = self.obsPtsLines = None

        # print('IMG', img.shape)
        if imgs is not None:
            if len(self.imgTexs) > 0: glDeleteTextures(self.imgTexs)
            self.imgTexs = glGenTextures(len(imgs))
            for imgTex,img in zip(self.imgTexs,imgs):
                glBindTexture(GL_TEXTURE_2D, imgTex)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.shape[1], img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
                glBindTexture(GL_TEXTURE_2D, 0)


    def renderCam(self, animIdx, viewIdx):

        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glEnable(GL_BLEND)

        # Lines connecting points.
        glColor4f(1,0,1,.3)
        # pts1 = np.hstack((pts, ptsFar)).reshape(-1,4)
        pts1 = self.obsPtsLines[animIdx,viewIdx]
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(4,GL_FLOAT,0,pts1)
        glDrawArrays(GL_LINES,0,len(pts1)*2)
        glDisableClientState(GL_VERTEX_ARRAY)

        # Frustum.
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        mv = self.view @ self.camMats[animIdx,viewIdx]
        glLoadMatrixf(mv.T.reshape(-1))
        glColor4f(1,1,1,1)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3,GL_FLOAT,0,frustum_pts)
        glDrawElements(GL_LINES,frustum_inds.size,GL_UNSIGNED_SHORT,frustum_inds)
        glDisableClientState(GL_VERTEX_ARRAY)
        glPopMatrix()

        # Image.
        if len(self.imgTexs) > 0:
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            mv = self.view @ self.camMats[animIdx,viewIdx]
            glLoadMatrixf(mv.T.reshape(-1))
            glColor4f(1,1,1,1)
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.imgTexs[animIdx])
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
        # print(x[0]-x[1])

        glEnable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

        x[...,6] = self.opacity * .1

        if self.viewOne:
            # glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE);
            viewIdx = self.viewIdx % x.shape[0]
            x = np.copy(x)
            # x[...,6] = min(max(.01, .05 * (32 / x.shape[0])), .1)
            x[viewIdx,:,6] = 1.

            if self.cocoSkeletons3d is not None:
                glDisable(GL_DEPTH_TEST)
                glLineWidth(4)
                glColor4f(.9,.3,0,.3)
                cx = self.cocoSkeletons3d[I,viewIdx]
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(3,GL_FLOAT,0,cx)
                glDrawElements(GL_LINES,self.cocoInds.size,GL_UNSIGNED_SHORT,self.cocoInds)
                glDisableClientState(GL_VERTEX_ARRAY)
                glEnable(GL_DEPTH_TEST)
                glLineWidth(1)

            if self.camMats is not None:
                self.renderCam(I,viewIdx)


        else:
            # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA)

        glDisable(GL_DEPTH_TEST)

        # Render frame in bottom left corner
        if len(self.imgTexs) > 0:
            cornerPts = np.array((
                -1,-1,0,
                 1,-1,0,
                 1, 1,0,
                -1, 1,0),dtype=np.float32).reshape(-1,3)
            cornerUvs = np.array((
                 0,1,
                 1,1,
                 1,0,
                 0,0),dtype=np.float32).reshape(-1,2)
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            ratio = (self.wh[1]/self.wh[0]) / (self.imgWH[1] / self.imgWH[0])
            size = .4 # X% of screen
            glOrtho(
                    # -1, (-1+size),
                    # -1, (-1+size*ratio),
                    -1, -1 + 2/(size*ratio),
                    -1, -1 + 2/(size),
                    0, 99)
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            animIdx = self.animationTime
            # mv = self.view @ self.camMats[animIdx,viewIdx]
            # glLoadMatrixf(mv.T.reshape(-1))
            glLoadIdentity()
            glDisable(GL_CULL_FACE)
            glColor4f(1,1,1,1)
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.imgTexs[animIdx])
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            glVertexPointer(3,GL_FLOAT,0,cornerPts)
            glTexCoordPointer(2,GL_FLOAT,0,cornerUvs)
            glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_SHORT,frustum_face_inds)
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            glBindTexture(GL_TEXTURE_2D, 0)
            glDisable(GL_TEXTURE_2D)
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)

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
        if key == '=': self.opacity *= 1.25
        if key == '-': self.opacity /= 1.25
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

    def sample(self, imgs0, B=64, T=40):
        with torch.no_grad():
            N = imgs0.shape[0]
            S = self.basePose.size(0)
            # print(x)

            camFxy = torch.FloatTensor((600,600)).cuda()
            camWh = torch.FloatTensor((512,512)).cuda()

            # Extract coco skeleton from the image.
            assert imgs0.shape[1] == 512
            assert imgs0.shape[2] == 512
            assert imgs0.shape[3] == 3

            all_xs = []
            all_skelPts = []

            for imgi, img in enumerate(imgs0):
                if imgi % 10 == 0:
                    print(f' - Image {imgi} / {len(imgs0)}')
                x = self.basePose.view(1,S) + torch.randn(B,S,device=self.basePose.device)
                out, vimg = run_model_and_viz(self.model2d, img, show=False)
                img[:] = vimg

                keypoints = out['keypoints'][0]
                # print('kpts shape', out['keypoints'].shape)
                # print('kpts shape', keypoints.shape)
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
                base = (self.sigma0/self.sigma1) ** (1./T)
                all_sigs = [self.sigma1 * (base ** t) for t in range(T)]
                all_sigs = torch.cuda.FloatTensor(all_sigs)
                print(f' - going from sigma {all_sigs[0]:.3f} to {all_sigs[-1]:.3f}')


                NOISE_WEIGHT = .3
                MODEL_WEIGHT = .9

                print(' - Doing Reverse Diffusion Process...')

                for i in range(T):
                    ts  = torch.cuda.FloatTensor([1. - i/T]).view(1).repeat(B)
                    sig = all_sigs[i].view(1).repeat(B)
                    # print(x.shape, ts.shape, z.shape)
                    ts = ts.view(-1,1)
                    z = z.flatten(1)
                    s = self.model3d(x,ts,z)

                    new_randomness = torch.randn_like(s) * sig.view(B,1) * NOISE_WEIGHT

                    # Add!
                    x += s * MODEL_WEIGHT + new_randomness

                print(' -                                ... Done')

                all_xs.append(x)
                all_skelPts.append(estSkelPts)

            BB = all_xs[0].size(0)

            estSkelPts = torch.stack(all_skelPts,0)

            # [N,BB,L,3] tensor
            xs = torch.stack(all_xs,0).reshape(N, BB, -1, 3)
            L = xs.shape[-2]
            xs_numpy = xs.cpu().numpy()

            inds0 = self.ppDset.inds
            inds = np.concatenate([inds0+L*i for i in range(BB)])


            if 1:
                cocoSkeletons3d = self.to_coco(xs.reshape(-1, L*3)).reshape(N, BB, -1, 3).cpu().numpy()

            if 1:
                print(estSkelPts.shape)
                obspts = estSkelPts.cpu().view(N*BB,-1,2)

                camPoses = torch.FloatTensor((1,0,0,0, 0,0,-3)).view(1,7).repeat(BB,1)

                # q0=torch.FloatTensor((1,0,0,0))
                q0=torch.FloatTensor((0,1,0,0))
                t0=torch.FloatTensor((0,1,2))

                # Use last time step only for now
                print(' - Recovering cameras...')
                print(xs.shape)
                print(cocoSkeletons3d.shape)
                print(obspts.shape)
                camPoses = recover_camera(camWh.cpu(), camFxy.cpu(), obspts,
                                          torch.from_numpy(cocoSkeletons3d).view(N*BB,-1,3),
                                          initialQ=q0,initialEye=t0).view(N,BB,-1)
                print(' -                   ... Done')

            renderer = LD_Renderer(1024,1364)
            renderer.init(True)
            print(' - Setting renderer data...')
            renderer.setData(inds, xs_numpy,
                             cocoSkeletons3d=cocoSkeletons3d, cocoInds=self.coco_inds,
                             camPosess = camPoses, fxy=camFxy.cpu(), wh=camWh.cpu(), zs=z.cpu(), imgs=imgs
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
    parser.add_argument('--video', required=True)
    parser.add_argument('-n', default=64, type=int)
    parser.add_argument('-s', '--stride', default=30, type=int)
    parser.add_argument('-T', default=30, type=int, help='diffusion steps')
    parser.add_argument('--numFrames', default=9999, type=int, help='max num frames')
    parser.add_argument('--skipFrames', default=0, type=int, help='skip num frames')
    args = parser.parse_args()

    meta = {}

    ppDset = PosePriorDataset(args.dsetFile, masterScale=.03)

    model3d,model3d_meta = get_model(dict(load=args.load))
    model3d = model3d.cuda().eval()

    model2d = FasterRcnnModel(get_resnet50().cuda().eval())
    vcap = cv2.VideoCapture(args.video)
    imgs = []
    for i in range(args.skipFrames): stat,f = vcap.read()
    while True:
        for i in range(args.stride): stat,f = vcap.read()
        # for i in range(120): stat,f = vcap.read()
        if stat <= 0: break
        f = f[...,[2,1,0]]
        f = crop_image(f)
        imgs.append(f)
        if len(imgs) >= args.numFrames: break
    imgs = np.stack(imgs)
    print(' - Images tensor shape', imgs.shape)

    sampler = Sampler(meta, model3d, model2d, ppDset, args.basePoseIdx)
    sampler.sample(imgs, B=args.n)

