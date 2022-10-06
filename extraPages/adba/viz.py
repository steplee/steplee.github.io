import cv2
import numpy as np, time
import torch, torch.nn, torch.nn.functional as F

def q_to_matrix1_(q):
    # r,i,j,k = q[0], q[1], q[2], q[3]
    # return torch.stack((
    r,i,j,k = q[0:1], q[1:2], q[2:3], q[3:4]
    return torch.cat((
        1-2*(j*j+k*k), 2*(i*j-k*r), 2*(i*k+j*r),
        2*(i*j+k*r), 1-2*(i*i+k*k), 2*(j*k-i*r),
        2*(i*k-j*r), 2*(j*k+i*r), 1-2*(i*i+j*j))).view(3,3)

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
    [0,1], [1,2], [2,3], [3,0],
    [4+0,4+1], [4+1,4+2], [4+2,4+3], [4+3,4+0],
    [0,0+4], [1,1+4], [2,2+4], [3,3+4],]).view(-1,2).cuda()

# glsl code
'''
float distanceToLineSegment(vec2 a, vec2 b, vec2 q, inout vec2 p, inout float t) {
    t = dot(q-a, normalize(b-a)) / (distance(a,b)+1e-6);
    t = clamp(t, 0., 1.);
    p = a + t * (b-a);
    return distance(p,q);
}
'''
# Using cv2.polylines makes much more sense.
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
    t = ((qa * ab.view(1,12,2)).sum(-1) / dab.view(1,12)).clamp(0,1)
    p = ps[:,1] + t.view(-1,12,1) * ab1

    dist = (p - uv.view(-1,1,2)).norm(dim=-1).min(1).values
    alpha = torch.exp(-dist * 750)
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
        self.frame = torch.zeros((512,512,3),dtype=torch.int32).cuda().contiguous()

        self.pts = (torch.rand(100,3).cuda() * 2 - 1) * 5

        # self.R = torch.diag(torch.tensor([1.,-1.,-1.])).cuda()
        # self.t = torch.tensor([0,0,5.]).cuda()
        self.R = torch.diag(torch.tensor([1.,1.,1.])).cuda()
        self.t = torch.tensor([0,0,-12.]).cuda()

        self.uv = torch.stack(torch.meshgrid(
            torch.linspace(-1,1,512),
            torch.linspace(-1,1,512)), -1).cuda()[...,[1,0]]

        self.project_frustum = compile_frustum_()

        self.wait = 0

        # R, t, f, color
        if 0:
            R1 = torch.from_numpy(cv2.Rodrigues(np.array((.1,.2,.3)))[0]).cuda().float()
            self.poses = [
                    (   torch.eye(3).cuda(), torch.tensor([0,0,-2.]).cuda(), torch.ones(2).cuda(), torch.rand(4).cuda()),
                    (R1@torch.eye(3).cuda(), torch.tensor([2,0,-2.]).cuda(), torch.ones(2).cuda(), torch.rand(4).cuda()),
                        ]

    def set_data(self, pts, poses, fs, colors):
        self.poses = []
        for pose,f,col in zip(poses,fs,colors):
            r = q_to_matrix1_(pose[:4])
            t = (pose[4:])
            self.poses.append((r,t,f,col))
        self.pts = pts

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
            d = (-d * 600).exp()
            self.frame += (d * 255).clip(0,255).to(torch.int32)

            # Draw frustums 
            for pose in self.poses:
                pr,pt,f,col = pose
                # alpha = project_frustum_(pr,pt, f, R,t, uv)
                alpha = self.project_frustum(pr,pt, f, R,t, uv).view(-1,1)
                alpha = ((col[:3]*col[3]).view(1,3)*alpha*255).clip(0,255).to(torch.int32)
                self.frame += alpha.view(h,w,3)

            f = self.frame.clip(0,255).byte().cpu().numpy()
            f = np.copy(f,'C')
            print(' - draw took', (time.time()-tt)*1000, 'ms')


            f = cv2.resize(f,(0,0),fx=2,fy=2)
            cv2.imshow('frame', f[...,[2,1,0]])

            key = (cv2.waitKey(self.wait))
            if key > 0 and key < 512:
                key = chr(key)
            else: key = '-'
            v = torch.zeros_like(self.t)
            r = np.zeros(3,dtype=np.float32)
            if key == 'a': v[0] -= .1
            if key == 'd': v[0] += .1
            if key == 'w': v[2] += .1
            if key == 's': v[2] -= .1
            if key == 'q': r[2] += .02
            if key == 'e': r[2] -= .02
            dR = cv2.Rodrigues(r)[0]
            self.t += self.R @ v
            self.R = self.R @ torch.from_numpy(dR).cuda()
            return key, f


if __name__ == '__main__':
    viz = Viz()
    for i in range(1000):
        viz.draw()
