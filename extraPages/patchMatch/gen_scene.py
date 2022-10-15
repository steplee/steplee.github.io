import torch, torch.nn.functional, numpy as np

def fract(x): return x - torch.floor(x)

def rand33(x):
    a,b,c = x.view(-1,3).T
    return torch.stack((
        fract(.3+(29.3*a+1.7*b-c).sin()),
        fract(.5+(29.3*a+9.7*b-13.3*c).cos()),
        fract(.2+(19.3*a+19.7*b-23.3*c).sin())), -1)

F = .5
def boxes(x):
    y = x * F
    g = y.floor()
    ctr = x + rand33(g) * .5 + .5
    siz = rand33(g*1.33 + 40) * .5 + .3
    q = (abs(x-ctr) - siz)
    return q.clip(0,999999).norm(dim=-1) + torch.clip(q.max(1).values, -9999999, 0)

def box(x,ctr,siz):
    q = (abs(x-ctr) - siz)
    return q.clip(0,999999).norm(dim=-1) + torch.clip(q.max(1).values, -9999999, 0)

def sphere(x,ctr,r):
    return (x-ctr).norm(dim=-1) - r

def sdf(x):
    # d = boxes(x)
    d = torch.maximum(boxes(x), -x[...,2])
    # d[x.norm(dim=-1) < 5] = 5 - x.norm(dim=-1)
    # d = x.norm(dim=-1) - 1
    # d = torch.maximum(x.norm(dim=-1)-1, -x[...,2])

    # ctr = torch.ones(1,3).cuda() * 1
    # siz = torch.ones(1,3).cuda() * .1
    # d = torch.maximum(box(x,ctr,siz), -x[...,2])
    # d = box(x,ctr,siz)

    return d

def raymarch(ro, rd):
    h,w = ro.shape[:2]
    t = torch.zeros_like(ro[...,:1])

    for i in range(40):
        p = ro+t*rd
        d = sdf(p.view(-1,3)).view(h,w,1)
        t += d * .9

    pt = ro + t*rd

    # col = 255 * abs(pt / pt.norm(dim=-1,keepdim=True))
    col =  255 * rand33((F*pt).floor()).view(h,w,3) / (pt.norm(dim=-1,keepdim=True) / 10)

    return t, pt, col

B = 50
S = 10

torch.manual_seed(0)
boxes_ = torch.rand(B,2,3).cuda()
boxes_[:,0,0:2] -= .5
boxes_[:,0] *= 10
boxes_[:,1] *= .9

boxes_[0:3,0,2] += 23
boxes_[0:3,1] *= 14.9

# xyzr
spheres = torch.rand(S,4).cuda()
spheres[:,0:3] -= .5
spheres[:,:3] *= 10
spheres[:,3:] *= .9

boxColors = torch.rand(B,3).cuda()
sphereColors = torch.rand(S,3).cuda()

def raymarch2_(ro,rd):
    # Generate random boxes_


    h,w = ro.shape[:2]
    t = torch.zeros_like(ro[...,:1])

    for i in range(190):
        p = (ro+t*rd).view(-1,3)

        d = torch.ones_like(p[...,0])*99999
        for j in range(B):
            d = torch.min(d, box(p, boxes_[j,0:1], boxes_[j,1:2]))
        for j in range(S):
            d = torch.min(d, sphere(p, spheres[j,0:3], spheres[j,3:]))

        # d = sdf(p.view(-1,3)).view(h,w,1)
        d = d.view(h,w,1)
        t += d * .9999

    t = t.clip(0,100)
    return t

def raymarch2(ro,rd):
    h,w = ro.shape[:2]
    t = raymarch2_(ro,rd)

    # TODO get normal

    pt = ro + t*rd
    p = pt.view(-1,3)


    col = torch.zeros_like(ro)
    d = t.clone()
    d[:] = .01 # larger distance than this = be black
    for j in range(B):
        dd = box(p, boxes_[j,0], boxes_[j,1]).view(h,w,-1)
        tmp = torch.stack((d,dd), 0).min(0)
        col = col*(tmp.indices==0) + boxColors[j]*(tmp.indices==1)
    for j in range(S):
        dd = sphere(p, spheres[j,:3], spheres[j,3]).view(h,w,-1)
        tmp = torch.stack((d,dd), 0).min(0)
        col = col*(tmp.indices==0) + sphereColors[j]*(tmp.indices==1)
    # col =  255 * col / (pt.norm(dim=-1,keepdim=True) / 10)
    col = 255 * col / ((4 + pt.norm(dim=-1,keepdim=True)) / 8)

    return t, pt, col

def gen_pair(wh, f, dt):
    w,h = wh
    ro1 = torch.zeros((w,h,3)).cuda()
    ro1[...,2] = -8
    ro2 = ro1 + dt.cuda()

    rd1 = torch.stack((*torch.meshgrid(
        torch.linspace(-1,1,w),
        torch.linspace(-1,1,h)),
        torch.ones(w,h)), -1).cuda()[...,[1,0,2]]
    # rd1[:,:,0] *= f[0] / (w*.5)
    # rd1[:,:,1] *= f[1] / (h*.5)
    rd1[:,:,0] *= (w*.5) / f[0]
    rd1[:,:,1] *= (h*.5) / f[1]
    rd1 = rd1 / rd1.norm(dim=-1,keepdim=True)
    rd2 = rd1.clone()

    depth1,pt1,col1 = raymarch2(ro1, rd1)
    depth2,pt2,col2 = raymarch2(ro2, rd2)

    img1 = col1.clip(0,255).to(torch.uint8)
    img2 = col2.clip(0,255).to(torch.uint8)

    return depth1, img1, depth2, img2

import cv2
dt = torch.tensor([.5,0.,0.])
h,w = 512,512
fov = 70.
f = ((w/2) / np.tan(fov/2), (h/2) / np.tan(fov/2))
d1,i1,d2,i2 = gen_pair((512,512), f, dt)
print(d1.min(),d1.max())
cv2.imshow('img1', i1.cpu().numpy())
cv2.imshow('img2', i2.cpu().numpy())
cv2.imshow('dep1', (d1/d1.max()).cpu().numpy())
cv2.waitKey(0)
torch.save({
    'f': f,
    'dt': dt,
    'imgs': [i1,i2],
    'deps': [d1,d2]}, 'out/scene1.pt')
