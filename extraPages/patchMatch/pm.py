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

def grid_for(h,w, dev, homog=False):
    # TODO verify grid alignment and such ...
    if homog:
        grid = torch.stack((*torch.meshgrid(
            torch.linspace(-1,1,w, device=dev),
            torch.linspace(-1,1,h, device=dev)),
            torch.ones((h,w),device=dev)), -1)[...,[1,0,2]]
    else:
        grid = torch.stack(torch.meshgrid(
            torch.linspace(-1,1,w, device=dev),
            torch.linspace(-1,1,h, device=dev)), -1)[...,[1,0]]
    return grid

'''
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
'''
def resampleWithGrid(i, grid):
    C,H,W = i.shape
    gh,gw = grid.shape[:2]
    grid = grid.view(1, gh,gw, 2)
    return F.grid_sample(i.view(1,C,H,W), grid).view(C,gh,gw).squeeze_()

def resampleWithPlane(a, b, fa, fb, pia):
    # TODO: Inputs must mathc?

    '''
    C,H,W = a.shape
    ga = grid_for(W,H, a.device, homog=True)
    gb = grid_for(W,H, a.device, homog=False)
    disp = (pia*ga).sum(-1)
    # print('pia',pia)
    # print('ga',ga)
    print('disparity * w',disp*a.shape[1])
    gb[...,0] -= disp
    # return resampleWithGrid(a, gb)
    return resampleWithGrid(b, gb)

    '''

    K = 7
    H,W = a.shape
    assert W == H
    # ga = (grid_for(W*K,H*K, a.device, homog=True) * W * .5 + W*.5).round()
    # ga = ga.reshape
    ga = (grid_for(W,H, a.device, homog=False) * W * .5 + W*.5).round()#.permute(1,0,2)
    dg = (grid_for(K,K, a.device, homog=False) * K * .49995).round()#.permute(1,0,2)
    ga = ga.view(H,W,1,1, 2) - dg.view(1,1,K,K, 2) # HWKK2
    ga = ga / W
    disp = (ga * pia[...,:2].view(H,W,1,1,2)).sum(-1) + pia[...,2].view(H,W,1,1) # HWKK
    ga[...,0] -= disp
    # print('ga', ga[256,256])

    # ga = ga.permute(1,3,0,2,4).reshape(H*K, W*K, 2)
    print('disparity * w at 256', disp[256,256]*W)
    # print('ga * w at 256', ga[256,256]*W)
    ga = ga.permute(0,2,1,3,4).reshape(H*K, W*K, 2)
    print('ga',ga.shape)
    aa = resampleWithGrid(a.view(1,H,W),ga).view(H,K,W,K)
    # aa = aa.permute(2,0,3,1).reshape(W,H,K*K).mean(-1)
    # aa = aa.permute(2,0,3,1).reshape(H,W,K*K)
    aa = aa.permute(0,2,1,3).reshape(H,W,K*K)
    return aa







# I don't quite understand the planar transfer, but I'll just keep going...
def photoError_sse(a,b, fa, fb, pia, K=7):
    h,w = b.size()
    # aa = resampleWithTranslationX(a, tx/w)
    # ba = resampleWithPlane(a,b,fa,fb, pia)
    # bau = F.unfold(ba.view(1,1,h,w), (K,K), padding=K//2).view(K*K,h,w).permute(1,2,0)
    # bu = F.unfold(b.view(1,1,h,w), (K,K), padding=K//2).view(K*K,h,w).permute(1,2,0)

    if 0:
        # Fronto-parallel
        abu = resampleWithPlane(a,b,fa,fb, pia)
        # abu = F.unfold(ab.view(1,1,h,w), (K,K), padding=K//2).view(K*K,h,w).permute(1,2,0)
        au = F.unfold(b.view(1,1,h,w), (K,K), padding=K//2).view(K*K,h,w).permute(1,2,0)
        d = abs(abu - au).mean(-1)
    else:
        # General plane
        # bu = F.unfold(b.view(1,1,h,w), (K,K), padding=K//2).view(K*K,h,w)
        # abu = resampleWithPlane(a.view(1,h,w),bu,fa,fb, pia)
        abu = resampleWithPlane(a,b,fa,fb, pia)
        print(abu[...,21])
        showTensor(abu[...,21],'abu21',-1)
        au = F.unfold(b.view(1,1,h,w), (K,K), padding=K//2).view(K*K,h,w).permute(1,2,0)
        d = abs(abu - au).mean(-1)


    print(d.shape)
    # showTensor(ba,'ba',-1)
    showTensor(a,'a',-1)
    showTensor(b,'b',-1)
    showTensor(d)


class PatchMatch:
    def __init__(self, sceneFile):
        self.scene = torch.load(sceneFile)
        # self.depthRange = (.001,.002)
        # self.depthRange = (.00001,.002)
        # self.depthRange = (1,1.1)
        self.depthRange = (.001,.01)
        # self.depthRange = (.000001,.00001)
        self.dev = torch.device('cuda')

    def random_init(self, h,w):
        D = self.dev

        z = torch.rand((h,w,1),device=D) * (self.depthRange[1]-self.depthRange[0]) + self.depthRange[0]
        # z[:] = 2
        z[:] = .0001
        # z /= h
        p = torch.cat((grid_for(h,w,dev=D), z), -1)
        # p[...,0] *= w
        # p[...,1] *= h

        n = torch.rand((h,w,3), device=D) # TODO:Transform for better distribution?
        n[...,:2] = (n[...,:2] - .5) * 2.
        n[...,2] += 20
        n = n / n.norm(dim=-1,keepdim=True)
        n[...,2] *= -1
        # n[...,0] *= -w / n[...,2]
        # n[...,1] *= -h / n[...,2]

        c = (n*p).sum(-1) / n[...,2]
        n[...,0] *= -1. / n[...,2]
        n[...,1] *= -1. / n[...,2]
        n[...,2]  = c
        return n

    def runOnPair(self):
        D = self.dev
        a,b = self.scene['imgs']
        a,b = prepImg(a), prepImg(b)
        dt = self.scene['dt']
        h,w = a.shape
        fa,fb = self.scene['f'], self.scene['f']

        # Actually, a normal is more of a 2d quantity...
        nrl = abs(torch.rand(h,w,3,device=D)) * torch.tensor([1,1,-1],device=D)
        dep = torch.rand(h,w,1,device=D)

        pia = self.random_init(h,w)
        pib = self.random_init(h,w)
        print(pia)

        # tx = torch.ones((h,w), device=D) * 40
        # photoError_sse(a,b, tx)
        photoError_sse(a,b, fa,fb, pia)


pm = PatchMatch('out/scene1.pt')
pm.runOnPair()
