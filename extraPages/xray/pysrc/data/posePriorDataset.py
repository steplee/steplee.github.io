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

class PosePriorDataset(Dataset):
    def __init__(self, file, masterScale=1):

        d = np.load(file)
        self.data = torch.from_numpy(d['data']) * masterScale
        self.data = self.data.reshape(self.data.shape[0], -1) # Flatten
        self.inds = torch.from_numpy(d['inds'].astype(np.int16))
        self.joints = {k:i*1 for (i,k) in enumerate(d['joints'])}

        if CENTER_DATASET:
            self.data = center_dataset(self.data, self.joints, self.inds)


        print(f' - PosePriorDataset(d={self.data.shape}, inds={self.inds.shape})')
        print(self.joints)

    def getStateSize(self): return self.data.shape[-1]

    def __len__(self):
        return len(self.data) * 2

    def __getitem__(self,i):
        if i >= len(self.data):
            x = self[i-len(self.data)]
            x[...,0] *= -1
            return x
        return self.data[i]


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

    # for i,p in enumerate(data[::5]):
    for i,p in enumerate(data):

        app.startFrame()
        app.render()

        if 1:
            glBegin(GL_LINES)
            glColor4f(1,0,0,1); glVertex3f(0,0,0); glVertex3f(1,0,0);
            glColor4f(0,1,0,1); glVertex3f(0,0,0); glVertex3f(0,1,0);
            glColor4f(0,0,1,1); glVertex3f(0,0,0); glVertex3f(0,0,1);
            glEnd()

        glColor4f(1,1,1,1)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3,GL_FLOAT,0,p)
        glDrawElements(GL_LINES,inds.size,GL_UNSIGNED_SHORT,inds)
        glDisableClientState(GL_VERTEX_ARRAY)

        if i % 1000 == 0:
            print('{:>7d}/{:>7d}'.format(i,len(data)))

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
