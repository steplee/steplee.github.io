import torch
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np

def showTensor(x, name='x', wait=0):
    import cv2
    if x.ndim == 2:
        x = (x.float().div(x.max()) * 255).clip(0,255).to(torch.uint8).cpu().numpy()
        x = np.copy(x,'C')
    else:
        assert False
    cv2.imshow(name,x)
    if wait >= 0: cv2.waitKey(wait)

def prepImg(x):
    assert x.dtype == torch.uint8
    if x.ndim == 3: x = x.float().mean(-1)
    x = x.float() / 255.
    return x.cuda()

def resampleWithTranslationX(a, t):
    h,w = a.shape
    # TODO verify grid alignment and such ...
    grid = torch.stack(torch.meshgrid(
        torch.linspace(-1,1,w),
        torch.linspace(-1,1,h)), -1).cuda()[...,[1,0]]
    print(grid.shape)
    grid[...,0] += t
    grid = grid.view(1, h,w, 2)
    return F.grid_sample(a.view(1,1,h,w), grid).view(h,w)

def photoError_sse(a,b, tx, K=3):
    h,w = b.size()
    aa = resampleWithTranslationX(a, tx/w)
    aau = F.unfold(aa.view(1,1,h,w), (K,K), padding=K//2).view(K*K,h,w).permute(1,2,0)
    bu = F.unfold(b.view(1,1,h,w), (K,K), padding=K//2).view(K*K,h,w).permute(1,2,0)

    d = abs(aau - bu).mean(-1)

    print(d.shape)
    showTensor(a,'aa',-1)
    showTensor(b,'b',-1)
    showTensor(d)


class PatchMatch:
    def __init__(self, sceneFile):
        self.scene = torch.load(sceneFile)

    def runOnPair(self):
        D = torch.device('cuda')
        a,b = self.scene['imgs']
        a,b = prepImg(a), prepImg(b)
        dt = self.scene['dt']
        h,w = a.shape

        # Actually, a normal is more of a 2d quantity...
        nrl = abs(torch.rand(h,w,3,device=D)) * torch.tensor([1,1,-1],device=D)
        dep = torch.rand(h,w,1,device=D)

        tx = torch.ones((h,w), device=D) * 40
        photoError_sse(a,b, tx)


pm = PatchMatch('out/scene1.pt')
pm.runOnPair()
