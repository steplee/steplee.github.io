import cv2
import numpy as np, time
import torch, torch.nn, torch.nn.functional as F

'''
A line and point renderer to go with ba stuff
'''

near = .2
far = 1
_f_pts = torch.tensor([
    -1,  1, 1,
     1,  1, 1,
     1, -1, 1,
    -1, -1, 1,
    -1,  1, 1,
     1,  1, 1,
     1, -1, 1,
    -1, -1, 1]).reshape(-1,3).cuda().float()
_f_pts[:4] *= near
_f_pts[4:] *= far
_inds = torch.tensor([
    [0,1],
    [1,2],
    [2,3],
    [3,0],
    [4+0,4+1],
    [4+1,4+2],
    [4+2,4+3],
    [4+3,4+0],
    [0,0+4],
    [1,1+4],
    [2,2+4],
    [3,3+4],]).view(-1,2).cuda()

'''
glsl code
float distanceToLineSegment(vec2 a, vec2 b, vec2 q, inout vec2 p, inout float t) {
    t = dot(q-a, normalize(b-a)) / (distance(a,b)+1e-6);
    t = clamp(t, 0., 1.);
    p = a + t * (b-a);
    return distance(p,q);
}
'''
def project_frustum_(R,t,f, camR, eye, uv):
    fpts = _f_pts.clone()
    fpts[:, :2] *= f * 1
    fpts = (fpts @ R.T) + t    # Transform into world
    fpts = (fpts - eye) @ camR # Transform into camera
    ppts = fpts[:, :2] / fpts[:, 2:] # Project into camera frame

    ps = ppts[_inds]

    # uv   ::      S x 2
    # ppts ::      8 x 2
    # ps   :: 12 x 2 x 2
    uv = uv.view(-1,2)
    S = uv.size(0)

    dab = (ps[:,0] - ps[:,1]).norm(dim=-1,keepdim=True) + 1e-8
    print(dab.shape,ps[:,0].shape)
    ab1 = (ps[:,0] - ps[:,1])
    ab = ab1 / dab

    # S x 12 x 2
    qa = uv.view(S, 1, 2) - ps[:, 1].view(1, 12, 2)
    # t = (qa @ ab.T).clamp(0,1)
    t = ((qa * ab.view(1,12,2)).sum(-1) / dab.view(1,12)).clamp(0,1)
    p = ps[:,1] + t.view(-1,12,1) * ab1

    dist = (p - uv.view(-1,1,2)).norm(dim=-1).min(1).values
    # dist = abs(t).min(1).values
    alpha = torch.exp(-dist * 150)
    return alpha


def compile_frustum_():
    R = torch.eye(3).cuda()
    camR = torch.eye(3).cuda()
    t = torch.rand(3).cuda()
    eye = torch.rand(3).cuda()
    f = torch.rand(2).cuda()
    uv = torch.stack(torch.meshgrid(
        torch.linspace(-1,1,512),
        torch.linspace(-1,1,512)), -1).cuda()
    project_frustum = torch.jit.trace(project_frustum_, (R,t,f, camR, eye, uv))
    return project_frustum


class Viz:
    def __init__(self):
        self.frame = torch.zeros((512,512,3),dtype=torch.uint8).cuda().contiguous()

        self.pts = (torch.rand(100,3).cuda() * 2 - 1) * 5

        self.R = torch.eye(3).cuda()
        self.t = torch.tensor([0,0,5.]).cuda()

        self.uv = torch.stack(torch.meshgrid(
            torch.linspace(-1,1,512),
            torch.linspace(-1,1,512)), -1).cuda()

        self.project_frustum = compile_frustum_()

        # R, t, f
        self.poses = [
                (torch.eye(3).cuda(), torch.tensor([0,0,-2.]).cuda(), torch.ones(2).cuda())
                    ]

    def draw(self):
        with torch.no_grad():

            tt = time.time()
            self.frame[:] = 0
            R,t,uv = self.R, self.t, self.uv
            h,w = self.frame.shape[:2]

            # self.frame += (uv * 255).clip(0,255).view(h,w,2).repeat(1,1,2)[...,:3].to(torch.uint8)

            # Draw points
            tpts = ((self.pts - self.t.view(1,3)) @ R) # N x 3
            ppts = tpts[...,:2] / tpts[...,2:] # N x 2
            n = ppts.size(0)
            if 0:
                d = (ppts.view(n, 1, 2) - uv.view(1, -1, 2)).norm(dim=-1) # N x S x 1
            else:
                d = torch.cdist(ppts,uv.view(-1,2))
            d = d.min(0).values.view(h,w,1)
            d = (-d * 1000).exp()
            self.frame += (d * 255).clip(0,255).to(torch.uint8)

            # Draw frustums 
            for pose in self.poses:
                pr,pt,f = pose
                # alpha = project_frustum_(pr,pt, f, R,t, uv)
                alpha = self.project_frustum(pr,pt, f, R,t, uv)
                alpha = (alpha*255).clip(0,255).to(torch.uint8)
                self.frame += alpha.view(h,w,1)

            f = self.frame.cpu().numpy()
            f = np.copy(f,'C')
            print(' - took', (time.time()-tt)*1000, 'ms')


            cv2.imshow('frame', f[...,[2,1,0]])
            cv2.waitKey(0)


if __name__ == '__main__':
    viz = Viz()
    for i in range(100):
        viz.t[0] += .01
        viz.draw()
