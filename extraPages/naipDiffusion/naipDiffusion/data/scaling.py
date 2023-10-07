import torch, torch.nn.functional as F, torch.nn as nn

class Scaler:
    # def __init__(self, K, C, sigma=1.5):
    def __init__(self):


        # Create gaussian blur for halfscale
        if 1:
            K = 5
            C = 3
            # sigma = 1.2
            sigma = .57

            x = torch.linspace(-K/2, K/2, steps=K)
            y = torch.linspace(-K/2, K/2, steps=K)
            grid = torch.cartesian_prod(y,x).cuda()
            d = grid.norm(dim=-1).pow(2)
            sigma2 = sigma*sigma
            f = (-d/(2*sigma2)).exp()
            f = f / (f.sum() * C) # Divind by C seems to be necessary too with groups.

            # We'll use a grouped conv, which means we don't need a 2d [C -> C] weight, but only a [C -> 1] one.
            f = f.view(1,1,K,K).repeat(C,1,1,1)
            self.C = C
            self.K = K
            print('halfscale kernel\n',f[0,0])
            self.halfscaleConvStride2 = nn.Conv2d(C,C,K,stride=2,padding=K//2,padding_mode='replicate',bias=False).cuda()
            self.halfscaleConvStride2.requires_grad_(False)
            self.halfscaleConvStride2.weight.data[:] = f

            self.halfscaleConvStride1 = nn.Conv2d(C,C,K,stride=1,padding=K//2,padding_mode='replicate',bias=False).cuda()
            self.halfscaleConvStride1.requires_grad_(False)
            self.halfscaleConvStride1.weight.data[:] = f

    def halfscale(self, x):
        with torch.no_grad():
            # Each of these gets nearly the same error on the checkerboard pattern
            return self.halfscaleConvStride2(x)
            # return F.avg_pool2d(self.halfscaleConvStride1(x), 2,2)
            # return self.halfscaleConvStride1(x)[:,:,::2,::2]

    def doublescale(self, x):
        with torch.no_grad():
            return F.interpolate(x, (x.size(-2)*2, x.size(-1)*2), mode='bilinear')

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False, linewidth=200)

    v = torch.zeros((1,3,256,256)).cuda()
    for y in range(4):
        for x in range(4):
            if y % 2 == 1 - x % 2:
                v[:,:, y*2*32:(y+1)*2*32, x*2*32:(x+1)*2*32] = 1

    import cv2, numpy as np

    vv = v.cpu().numpy()[0].transpose(1,2,0)
    vv = (vv * 255).clip(0,255).astype(np.uint8)
    cv2.imshow('v0',vv)

    s = Scaler()
    y = s.halfscale(v)

    yy = y.cpu().numpy()[0].transpose(1,2,0)
    yy = (yy * 255).clip(0,255).astype(np.uint8)
    cv2.imshow('y',yy)

    v2 = s.doublescale(y)

    vv2 = v2.cpu().numpy()[0].transpose(1,2,0)
    vv2 = (vv2 * 255).clip(0,255).astype(np.uint8)
    cv2.imshow('v2',vv2)

    e = abs(v-v2).cpu().numpy()[0].transpose(1,2,0)
    print('error sum', np.sum(e))
    e = (e * 255).clip(0,255).astype(np.uint8)
    cv2.imshow('err',e)
    cv2.waitKey(0)
