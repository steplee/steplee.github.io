import torch
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np

def showTensor(x, name='x', wait=0):
    import cv2
    if x.ndim == 2:
        x = (x.float().div(x.max()) * 255).clip(0,255).to(torch.uint8).cpu().numpy()
        x = np.copy(x,'C')
    else:
        assert False
    while x.shape[0] > 1080: x = cv2.resize(x, (0,0), fx=.5, fy=.5)
    cv2.imshow(name,x)
    if wait >= 0: cv2.waitKey(wait)

def prepImg(x):
    assert x.dtype == torch.uint8
    if x.ndim == 3: x = x.float().mean(-1)
    x = x.float() / 255.
    return x.cuda()

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

def photoError_sse(a,b, disp, K=7):
    h,w = b.size()

    g = grid_for(h,w, a.device)
    print(a.shape,b.shape,g.shape)
    g[...,0] += disp / w

    bu  = F.unfold(b.view(1,1,h,w), (K,K), padding=K//2).view(K*K,h,w)
    print('bu',bu.shape)
    print('g', g.shape)
    print('disp', g.shape)
    abu = resampleWithGrid(bu, g)

    au  = F.unfold( b.view(1,1,h,w), (K,K), padding=K//2).view(K*K,h,w)
    print(abu.shape,au.shape)
    d = abs(abu - au).mean(0)

    print(d.shape)
    # showTensor(ba,'ba',-1)
    showTensor(a,'a',-1)
    showTensor(b,'b',-1)
    showTensor(d)

class SimpleDisp:
    def __init__(self, scene):
        # self.scene = torch.load(sceneFile)
        self.dispRange = (0,30)
        self.scene = scene
        self.dev = torch.device('cuda')

    def random_init(self, h,w):
        D = self.dev
        d = torch.randint(self.dispRange[0],self.dispRange[1], (h,w), device=D) + self.dispRange[0]
        return d

    def runOnPair(self):
        D = self.dev
        a,b = self.scene['imga'], self.scene.imgb
        a,b = prepImg(a), prepImg(b)
        # dt = self.scene['dt']
        fa,fb = self.scene['intrina'], self.scene['intrinb']
        h,w = a.shape

        dba = self.random_init(h,w)
        dab = self.random_init(h,w)

        photoError_sse(a,b, dba)

from scene import Scene
pm = SimpleDisp(Scene.load('data/sceneMike.pt'))
pm.runOnPair()
