import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

CENTER_DATASET = True
def center_dataset(x, joint2idx, inds):
    with torch.no_grad():
        B,S = x.size()

        x = x.view(B,S//3,3)


        '''
        offset = torch.zeros((B,3), device=x.device)
        offset[:,1] = torch.min(
                x[:,[joint2idx['LeftToeBase'],
                     joint2idx['LeftToeBase_end'],
                     joint2idx['RightToeBase'],
                     joint2idx['RightToeBase_end'],
                     joint2idx['RightFoot'],
                     joint2idx['RightFoot']], 1], 1).values
        '''
        offset = torch.zeros((B,3), device=x.device)
        offset[:,1] = torch.min(x[..., 1], 1).values


        # Apply offset based on spine to XZ axes
        # offset[:] += x[:,joint2idx['Spine']] * torch.FloatTensor([1,0,1]).view(1,3).to(x.device)
        offset[:] += (x[:,joint2idx['Spine']] +
                      x[:,joint2idx['Hips']] +
                      x[:,joint2idx['Head']]
                      ) * torch.FloatTensor([1,0,1]).view(1,3).to(x.device) / 3
        scale = \
                ((x[:, joint2idx['LeftFoot']] - x[:, joint2idx['LeftLeg']]).norm(dim=-1) + \
                (x[:, joint2idx['LeftUpLeg']] - x[:, joint2idx['LeftLeg']]).norm(dim=-1) + \
                (x[:, joint2idx['RightFoot']] - x[:, joint2idx['RightLeg']]).norm(dim=-1) + \
                (x[:, joint2idx['RightUpLeg']] - x[:, joint2idx['RightLeg']]).norm(dim=-1)) * .5
        scale = scale.view(B,1)

        x.sub_(offset.view(B,1,3))
        x.div_(scale.view(B,1,1))

        # If body appears turned-around, rotate 180deg in Y axis
        # We can test if arms are criss-crossed,
        # or we can check if the feet seem to point backward
        # It's very hard to point both feet backward without turning, so I think that is good to go with.
        # rotate_mask = (x[:,joint2idx['LeftArm'], 0] > x[:,joint2idx['RightArm'], 0]).float().view(B,1,1)
        z1 = F.normalize(x[:,joint2idx['LeftToeBase_end']] - x[:,joint2idx['LeftToeBase']], dim=1)
        z2 = F.normalize(x[:,joint2idx['RightToeBase_end']] - x[:,joint2idx['RightToeBase']], dim=1)
        rotate_mask = ((z1+z2)[:,2] < 0).float().view(B,1,1)
        x = x*torch.FloatTensor([-1,1,-1]).view(1,1,3).to(x.device)*(rotate_mask) + x*(1-rotate_mask)

        x = x.view(B,S)

        return x


# Given a cuda tensor, rotate the poses to create more diversity.
# This will yaw around the local mean!
def randomly_yaw_batch(x):
    with torch.no_grad():
        (B,S),d = x.size(),x.device
        L = S // 3
        x = x.view(B,L,3)
        mu = x.mean((0,1),keepdims=True)

        angles = torch.randn(B,device=d) * (0 / np.pi)
        c,s = torch.cos(angles), torch.sin(angles)

        R = torch.eye(3,device=d).unsqueeze_(0).repeat(B,1,1)
        R[:,0,0] = c
        R[:,2,2] = c
        R[:,0,2] = -s
        R[:,2,0] = s

        x -= mu
        x = x @ R.mT
        x += mu

        return x.view(B,S)

class PosePriorDataset(Dataset):
    def __init__(self, file, masterScale=1):

        d = np.load(file)
        self.data = torch.from_numpy(d['data']) * masterScale
        self.data = self.data.reshape(self.data.shape[0], -1) # Flatten
        self.inds = torch.from_numpy(d['inds'].astype(np.int16))
        self.joints = {k:i*1 for (i,k) in enumerate(d['joints'])}

        if CENTER_DATASET:
            print(' - Centering dataset ...')
            self.data = center_dataset(self.data, self.joints, self.inds)
            print(' -                   ... Done')


        print(f' - PosePriorDataset(d={self.data.shape}, inds={self.inds.shape})')
        from pprint import pprint
        pprint(self.joints)

    def getStateSize(self): return self.data.shape[-1]

    def __len__(self):
        return len(self.data) * 2

    def __getitem__(self,i):
        if i >= len(self.data):
            x = self[i-len(self.data)]
            x[...,0] *= -1
            return x
        return self.data[i]


# def normalize(x): return x / np.linalg.norm(x)
def normalize(x): return x / (x*x).sum(1,keepdim=True).sqrt_()

class Map_PP_to_Coco_v1():
    def __init__(self, cocoJoints,  joints):
        self.joints = joints
        self.cocoJoints = cocoJoints
        self.inputDims = len(joints)*3
        self.outputDims = len(cocoJoints)*3

        l_swiz = {
                # 'l_shoulder': 'LeftArm',
                'l_elbow': 'LeftForeArm',
                'l_hand': 'LeftHand',
                # 'l_hip': 'LeftUpLeg',
                'l_knee': 'LeftLeg',
                'l_foot': 'LeftFoot',
        }
        r_swiz = {(k.replace('l_', 'r_'), v.replace('Left','Right')) for k,v in l_swiz.items()}

        # swiz = {'nose': 'Head_end'}
        swiz = {}
        swiz.update(l_swiz)
        swiz.update(r_swiz)
        print(r_swiz)

        swizzle = [0,]*len(cocoJoints)*3
        for akey, bkey in swiz.items():
            aidx = cocoJoints[akey]
            bidx = joints[bkey]
            swizzle[aidx*3:(aidx+1)*3] = range(bidx*3,bidx*3+3)

        self.swizzle = torch.LongTensor(swizzle)
        print(swizzle)


    def __call__(self, x):
        y = x[:, self.swizzle]

        # FIXME: These are not quite correct.
        #        Need to come up with something better.
        #        The nose sometimes flips to go the wrong way ... need to contextualize with shoulders to help.

         # We still have some extra stuff.
        nose  = self.cocoJoints['nose']
        l_eye = self.cocoJoints['l_eye']
        l_ear = self.cocoJoints['l_ear']
        l_sho = self.cocoJoints['l_shoulder']
        l_hip = self.cocoJoints['l_hip']
        r_eye = self.cocoJoints['r_eye']
        r_ear = self.cocoJoints['r_ear']
        r_sho = self.cocoJoints['r_shoulder']
        r_hip = self.cocoJoints['r_hip']
        headEnd = self.joints['Head_end']
        head    = self.joints['Head']
        lshould = self.joints['LeftShoulder']
        rshould = self.joints['RightShoulder']
        larm = self.joints['LeftArm']
        rarm = self.joints['RightArm']
        lupleg = self.joints['LeftUpLeg']
        rupleg = self.joints['RightUpLeg']

        # z_ = normalize(x[headEnd*3:headEnd*3+3] - x[head*3:head*3+3])
        z_ = normalize(x[:,head*3:head*3+3] - .5*(x[:,lshould*3:lshould*3+3]+x[:,rshould*3:rshould*3+3]))
        Y = torch.FloatTensor((0,1,1)).to(x.dtype).view(1,3).to(x.device)
        # Y = np.array((0,1,0),dtype=np.float32)
        x_ = normalize(torch.cross(x[:,head*3:head*3+3], Y))
        y_ = normalize(torch.cross(x_,z_))

        headOffset = z_*.0  + y_*-.07

        y[:,nose*3:nose*3+3] = x[:,headEnd*3:headEnd*3+3] - z_ * .03 + y_*.05 + headOffset

        y[:,l_eye*3:l_eye*3+3] = x[:,headEnd*3:headEnd*3+3] + x_ * .04 + headOffset
        y[:,l_ear*3:l_ear*3+3] = x[:,headEnd*3:headEnd*3+3] + x_ * .091 - z_ * .06 - y_*.07 + headOffset

        y[:,r_eye*3:r_eye*3+3] = x[:,headEnd*3:headEnd*3+3] - x_ * .04 + headOffset
        y[:,r_ear*3:r_ear*3+3] = x[:,headEnd*3:headEnd*3+3] - x_ * .091 - z_ * .06 - y_*.07 + headOffset

        y[:,l_sho*3:l_sho*3+3] = x[:,larm*3:larm*3+3] - z_ * .09 - x_*.02
        y[:,r_sho*3:r_sho*3+3] = x[:,rarm*3:rarm*3+3] - z_ * .09 + x_*.02

        # FIXME: These need to move to be wider.
        y[:,l_hip*3:l_hip*3+3] = x[:,lupleg*3:lupleg*3+3]
        y[:,r_hip*3:r_hip*3+3] = x[:,rupleg*3:rupleg*3+3]

        return y


def glut_print(x,  y,  font,  text):
    glEnable(GL_BLEND)
    glColor3f(1,1,1)
    glPushMatrix()
    # glWindowPos2f(x,y)
    glRasterPos2f(x,y)
    for ch in text :
        glutBitmapCharacter( font , ctypes.c_int( ord(ch) ) )
    glPopMatrix()


def show_anim_3d_exported(fi):
    '''
    d = np.load(fi)
    data = d['data']
    inds = d['inds']
    '''
    dset = PosePriorDataset(fi, masterScale=.03)
    data,inds = dset.data, dset.inds

    app = SurfaceRenderer(1024,1024)
    app.init(True)

    inds = inds.cpu().numpy().astype(np.uint16)
    data = data.cpu().numpy().astype(np.float32)
    # data[:] *= .1

    to_coco = None
    if 1:
        from ..est2d.run import get_coco_skeleton
        coco_inds, cocoJoints = get_coco_skeleton()
        coco_inds = np.array(coco_inds,dtype=np.uint16)
        to_coco = Map_PP_to_Coco_v1(cocoJoints, dset.joints)

    # for i,p in enumerate(data[::5]):
    # for i,p in enumerate(data):

    i = 0
    while i < len(data):
        if app.n_pressed:
            i += 1
            if i >= len(data): break
            if i % 1000 == 0:
                print('{:>7d}/{:>7d}'.format(i,len(data)))
        p = data[i]

        app.startFrame()
        app.render()
        glLineWidth(1)


        # Show axes.
        if 1:
            glBegin(GL_LINES)
            glColor4f(1,0,0,1); glVertex3f(0,0,0); glVertex3f(1,0,0);
            glColor4f(0,1,0,1); glVertex3f(0,0,0); glVertex3f(0,1,0);
            glColor4f(0,0,1,1); glVertex3f(0,0,0); glVertex3f(0,0,1);
            glEnd()

        # Render PP skeleton
        glColor4f(1,0,1,1)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3,GL_FLOAT,0,p)
        glDrawElements(GL_LINES,inds.size,GL_UNSIGNED_SHORT,inds)
        glDisableClientState(GL_VERTEX_ARRAY)

        # Show COCO-mapped skeleton.
        if to_coco is not None:
            pp = torch.from_numpy(p).view(1,-1)
            coco_pts = to_coco(pp).numpy()[0]
            glLineWidth(4)
            glColor4f(0,1,.1,.5)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3,GL_FLOAT,0,coco_pts)
            glDrawElements(GL_LINES,coco_inds.size,GL_UNSIGNED_SHORT,coco_inds)
            glDisableClientState(GL_VERTEX_ARRAY)

        # Show text with PP joint names.
        if 1:
            V = glGetFloatv(GL_MODELVIEW_MATRIX)
            P = glGetFloatv(GL_PROJECTION_MATRIX)
            MVP = P.T@V.T
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            for name, idx in dset.joints.items():
                pp = MVP @ np.array((*p[idx*3:idx*3+3],1.))
                pp = pp[:3]/pp[3:]
                # glut_print(pp[0], pp[1], GLUT_BITMAP_9_BY_15, name)
                glut_print(pp[0], pp[1], GLUT_BITMAP_HELVETICA_10, name)


        app.endFrame()

if __name__ == '__main__':
    from ..render import *

    from argparse import ArgumentParser
    parser = ArgumentParser()
    # Show a 3d animation of already exported data
    parser.add_argument('--show', default='', required=True)
    args = parser.parse_args()

    if args.show:
        show_anim_3d_exported(args.show)
