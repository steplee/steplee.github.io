import torch
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from matplotlib.cm import bwr
import cv2

SHOW_SSE  = False
SHOW_DISP = False
SHOW_DISP = True

def scale_homography(H, s):
    return H * np.array((1,1,1/s, 1,1,1/s, s,s,1), dtype=H.dtype).reshape(3,3)

def apply_homog(H, xs):
    if xs.shape[-1] == 2:
        xs = np.concatenate((xs, np.ones_like(xs[...,:1])),-1)
    y = xs @ H.T
    return y[...,:2] / y[...,2:]

def showTensor(x, name='x', wait=0):
    import cv2
    if x.ndim == 2:
        x = (x.float().div(x.max()) * 255).clip(0,255).to(torch.uint8).cpu().numpy()
        x = np.copy(x,'C')
    else:
        assert x.size(-1) == 3
        x = (x.float() * 255).clip(0,255).to(torch.uint8).cpu().numpy()
        x = np.copy(x[...,[2,1,0]],'C')
        # x = np.copy(x[...,[0,1,2]],'C')

    while x.shape[0] > 1080: x = cv2.resize(x, (0,0), fx=.5, fy=.5)
    cv2.imshow(name,x)
    if wait >= 0: cv2.waitKey(wait)

def prepImg(x, scale=0):
    x = x.cuda()
    xc = x
    assert x.dtype == torch.uint8
    h,w = x.shape[:2]
    if x.ndim == 2 or x.size(-1) == 1: xc = xc.view(h,w,1).repeat(1,1,3)
    if x.ndim == 3: x = x.float().mean(-1)

    x = x.float() / 255.
    xc = xc.float() / 255.
    if scale != 0:
        x = F.interpolate(x.view(1,1,h,w), scale_factor=.5)[0,0]
        xc = F.interpolate(xc.permute(2,0,1).view(1,3,h,w), scale_factor=.5)[0].permute(1,2,0)
    return x, xc

def grid_for(h,w, dev, homog=False):
    # TODO verify grid alignment and such ...
    if homog:
        grid = torch.stack((*torch.meshgrid(
            torch.linspace(-1,1,h, device=dev),
            torch.linspace(-1,1,w, device=dev)),
            torch.ones((h,w),device=dev)), -1)[...,[1,0,2]]
    else:
        grid = torch.stack(torch.meshgrid(
            torch.linspace(-1,1,h, device=dev),
            torch.linspace(-1,1,w, device=dev)), -1)[...,[1,0]]
    return grid

def resampleWithGrid(i, grid):
    C,H,W = i.shape
    gh,gw = grid.shape[:2]
    grid = grid.view(1, gh,gw, 2)
    return F.grid_sample(i.view(1,C,H,W), grid).view(C,gh,gw).squeeze_()

# Allows passing a map to remap with
def photoError_sse_general(a,b, disp, K=7):
    h,w = b.size()

    g = grid_for(h,w, a.device)
    print(a.shape,b.shape,g.shape)
    g[...,0] += disp / w

    bu  = F.unfold(b.view(1,1,h,w), (K,K), padding=K//2).view(K*K,h,w)
    abu = resampleWithGrid(bu, g)

    au  = F.unfold( a.view(1,1,h,w), (K,K), padding=K//2).view(K*K,h,w)
    print(abu.shape,au.shape)
    d = abs(abu - au).mean(0)

    print(d.shape)
    # showTensor(ba,'ba',-1)
    # showTensor(a,'a',-1)
    # showTensor(b,'b',-1)
    if SHOW_SSE:
        showTensor(d, wait=1)

__kernels = {}

def photoError_sse_oneDisp(a,b, dispValue, K=25):
    with torch.no_grad():
        global __kernels
        h,w = b.size()
        # print(h,w)

        g = grid_for(h,w, a.device)
        g[...,0] += dispValue / w

        ab = resampleWithGrid(b.view(1,h,w), g).view(h,w)

        d = abs(ab - a)
        if 0:
            d = F.avg_pool2d(d.view(1,-1,h,w), K, padding=K//2, stride=1)
        else:
            if K not in __kernels:
                sigma = K * .4
                W = ((grid_for(K,K, a.device) * K * .5).pow(2).sum(-1) / (-2 * sigma*sigma)).exp().view(1,1,K,K)
                W = W.div_(W.sum())
                __kernels[K] = W
                if SHOW_SSE:
                    showTensor(W.view(K,K),'W', wait=1)
            else:
                W = __kernels[K]
            d = F.conv2d(d.view(1,1,h,w), W, padding=K//2).view(h,w)
        d = d.view(h,w)

        if SHOW_SSE:
            showTensor(d, wait=1)
        # showTensor(d, wait=0)

        return d

class SimpleDisp:
    def __init__(self, scene):
        # self.scene = torch.load(sceneFile)
        # self.dispRange = np.array((-90,30))
        self.dispRange = np.array((-100,100))
        # self.dispRange = np.array((-10,10))
        self.scene = scene
        self.dev = torch.device('cuda')

    def random_init(self, h,w):
        D = self.dev
        d = torch.randint(self.dispRange[0],self.dispRange[1], (h,w), device=D) + self.dispRange[0]
        return d

    def runOnPair(self):
        D = self.dev
        a,b = self.scene['imga'], self.scene.imgb
        scale = .5
        (a,ac), (b,bc) = prepImg(a,scale), prepImg(b,scale)
        # dt = self.scene['dt']
        h,w = a.shape

        # dba = self.random_init(h,w)
        # dab = self.random_init(h,w)

        # photoError_sse_general(a,b, dba)

        minVals, minDisp = None,None
        for d in torch.arange(*self.dispRange):
            e = photoError_sse_oneDisp(a,b, d)
            # e = photoError_sse_oneDisp(b,a, d)
            if minVals is None: minVals, minDisp = e, (e*0)+d
            else:
                mask = (e <= minVals).float()
                minDisp = d * mask + minDisp * (1-mask)
                minVals = torch.min(minVals, e)
            # print(d, e.sum().item())

        # print(minDisp)
        # print(dimg)

        if SHOW_DISP:
            dimg = (minDisp / max(abs(self.dispRange)) + 1) * .5
            dimg = torch.from_numpy(bwr(dimg.cpu().numpy()))[...,:3]
            showTensor(dimg, name='disp', wait=1)
        return minDisp, minVals

    def getPtsFromDispTwoD(self, disp, mask0, scale, isA=True):
        w,h = (self.scene.intrina.wh * scale).astype(int)

        Hinv = np.linalg.inv((self.scene.Ha if isA else self.scene.Hb).cpu().numpy())
        # print(' Hinv', Hinv)
        # print('Mapped corners before homog:\n',apply_homog(Hinv, np.array((0,0, w,0, w,h, 0,h)).reshape(4,2)))
        iscale = 1/scale
        Hinv = scale_homography(Hinv, iscale)
        Hbinv = scale_homography(np.linalg.inv(self.scene.Hb.cpu().numpy()), iscale)
        Hb = scale_homography((self.scene.Hb.cpu().numpy()), iscale)
        Ha = scale_homography((self.scene.Ha.cpu().numpy()), iscale)
        # print(' Hinv scaled', Hinv)
        # print('Mapped corners after homog:\n',apply_homog(Hinv, np.array((0,0, w,0, w,h, 0,h)).reshape(4,2)))

        if 0:
            pts1 = (grid_for(h,w, torch.device('cpu')).cpu().numpy() + 1) * (w,h)
            pts1 = np.concatenate((pts1, np.ones_like(pts1[...,:1])), -1) # make homogoneous

            pts2 = (grid_for(h,w, torch.device('cpu')).cpu().numpy() + 1) * (w,h)
            # pts2[...,1] += disp.cpu().numpy() / scale
            # pts2[...,0] += disp.cpu().numpy() / scale
            pts2 = np.concatenate((pts2, np.ones_like(pts2[...,:1])), -1) # make homogoneous

            pts1 = cv2.warpPerspective(pts1, Hinv, (w,h), flags=cv2.INTER_NEAREST)
            pts2 = cv2.warpPerspective(pts2, Hinv, (w,h), flags=cv2.INTER_NEAREST)
            mask = cv2.warpPerspective(mask0, Hinv, (w,h), flags=cv2.INTER_NEAREST)
        else:
            pts1 = (grid_for(h,w, torch.device('cpu')).cpu().numpy() + 1) * (w,h)
            pts1 = np.concatenate((pts1, np.ones_like(pts1[...,:1])), -1) # make homogoneous

            pts2 = (grid_for(h,w, torch.device('cpu')).cpu().numpy() + 1) * (w,h)
            # pts2[...,1] += disp.cpu().numpy() / scale
            # pts2 = cv2.warpPerspective(pts2, Hbinv @ Ha, (w,h), flags=cv2.INTER_NEAREST)
            pts2 = cv2.warpPerspective(pts2, Ha, (w,h), flags=cv2.INTER_NEAREST)
            pts2[...,0] -= disp.cpu().numpy() / scale
            pts2 = cv2.warpPerspective(pts2, Hbinv, (w,h), flags=cv2.INTER_NEAREST)
            pts2 = np.concatenate((pts2, np.ones_like(pts2[...,:1])), -1) # make homogoneous

            # pts1 = cv2.warpPerspective(pts1, Hinv, (w,h), flags=cv2.INTER_NEAREST)
            mask = cv2.warpPerspective(mask0, Hinv, (w,h), flags=cv2.INTER_NEAREST)

        if SHOW_DISP:
            dimg = ((disp / max(abs(self.dispRange)) + 1) * .5).cpu().numpy()
            print(dimg.shape, dimg.dtype)
            dimg = cv2.warpPerspective(dimg, Hinv, (w,h))
            dimg = torch.from_numpy(bwr(dimg))[...,:3]
            showTensor(dimg, name='warped disp', wait=1)

        return pts1, pts2, mask

    def runTriangulation(self, disp):
        from matchingTransformation import decompose_essential
        from triangulation import triangulate

        # This is wrong, I think I must inverse warp the disparity map back to original image (which is where F is computed)

        scale = .5
        a,b = self.scene['imga'], self.scene.imgb
        (a,ac), (b,bc) = prepImg(a,scale), prepImg(b,scale)
        intrina,intrinb = scene['intrina'], scene['intrinb']
        awh = intrina.wh
        KA = intrina.K()
        KB = intrinb.K()

        F = self.scene.F.cpu().numpy()
        u,s,vt = np.linalg.svd(F)
        # print(' - svd of F', u,s,vt)
        # print('det',np.linalg.det(u@vt))
        someWorldPoint = np.array((.000001,.000001,5.5)) # Is this correct? I think, because assume PA = I_4
        # E = KB.T @ F @ KA
        # decompose_essential(E)
        VB = decompose_essential(F, KA, KB, someWorldPoint)
        VA = np.eye(4)
        VB_ = np.eye(4); VB_[:3] = VB; VB = VB_


        if 0:
                # pts = triangulate(VA,VB, intrina, intrinb, disp)
                w,h = int(intrina[-2]*.5*scale), int(intrina[-1]*.5*scale)
                ptsa = (grid_for(h,w, torch.device('cpu')).cpu().numpy() + 1) * .5 * (w,h)
                ptsb = np.copy(ptsa)
                ptsb[...,1] += disp.cpu().numpy() / scale
                ptsa,ptsb = ptsa.reshape(-1,2), ptsb.reshape(-1,2)
        else:
                # w,h = int(intrina[-2]*scale), int(intrina[-1]*scale)
                w,h = int(awh[0]*scale), int(awh[1]*scale)
                mask0 = np.ones((int(awh[0]),int(awh[1])),dtype=np.float32)
                H = scale_homography(self.scene.Ha.cpu().numpy(), 1/scale)
                cv2.warpPerspective(mask0, H, ((w,h)))
                ptsa,ptsb,mask = self.getPtsFromDispTwoD(disp, mask0, scale)
                if 1:
                    p = torch.from_numpy(ptsa);
                    p = p.view(-1,3);
                    p = p - p.min(0).values.view(-1,3);
                    p = (p / (p.max(0).values.view(-1,3) + 1e-6)).view(h,w,3)
                    showTensor(p, 'ptsa', 1)
                    p = torch.from_numpy(ptsb);
                    p = p.view(-1,3);
                    p = p - p.min(0).values.view(-1,3);
                    p = (p / (p.max(0).values.view(-1,3) + 1e-6)).view(h,w,3)
                    showTensor(p, 'ptsb')
                ptsa = ptsa[...,:2].reshape(-1,2)
                ptsb = ptsb[...,:2].reshape(-1,2)
        pts = triangulate(VA,VB, KA, KB, ptsa,ptsb)

        a,ac = prepImg(self.scene.rawImga, scale)

        return pts, ac, mask, VA,VB,KA,KB

from scene import Scene, Intrinsics
# scene = Scene.load('data/sceneMike.pt')
# scene = Scene.load('data/sceneStrathmore.pt')
scene = Scene.load('data/sceneNight.pt')
# scene = Scene.load('data/scenePaper.pt')

if 1:
    pm = SimpleDisp(scene)
    # pm = SimpleDisp(Scene.load('data/sceneStrathmore.pt'))
    disp, dispErr = pm.runOnPair()
    pts, colors, mask, PA,PB,KA,KB = pm.runTriangulation(disp)
    # colors = colors.flip(0)
    colors = (colors.cpu().numpy().astype(np.float32))
    colors = colors * (mask[...,np.newaxis]>.5)
    colors = colors.reshape(-1,3)[...,[2,1,0]]
    colors = np.hstack((colors, np.ones_like(colors[:, -1:])))
    print(pts.shape, colors.shape)
    cv2.waitKey(1)

from viewer import Viewer
if __name__ == '__main__':
    viz = Viewer(1024,1024)
    viz.init(True)


    intrina,intrinb = scene['intrina'], scene['intrinb']
    # KA = np.array((intrina[0], 0, intrina[2], 0, intrina[1],intrina[3],0,0,1)).reshape(3,3)
    # KB = np.array((intrinb[0], 0, intrinb[2], 0, intrinb[1],intrinb[3],0,0,1)).reshape(3,3)
    wha,whb = intrina.wh, intrinb.wh
    print('shapes1',PA.shape,PB.shape)
    viz.setCamera(0, PA, KA, wha)
    viz.setCamera(1, PB, KB, whb)

    print(pts,colors)
    # colors[:,0] = 1
    viz.setPointCloud(pts, colors)

    for i in range(10000):
        viz.startFrame()
        viz.render()
        if viz.endFrame(): break
