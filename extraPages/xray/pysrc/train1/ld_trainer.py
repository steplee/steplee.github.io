import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..data.posePriorDataset import PosePriorDataset
from ..loss_dict import LossDict
import os, sys

from .models import *

# Need to map torhcvision joints to this dset joints.

def q_to_matrix(q):
    r,i,j,k = q[:,0], q[:,1], q[:,2], q[:,3]
    return torch.stack((
        1-2*(j*j+k*k),
        2*(i*j-k*r),
        2*(i*k+j*r),
        2*(i*j+k*r),
        1-2*(i*i+k*k),
        2*(j*k-i*r),
        2*(i*k-j*r),
        2*(j*k+i*r),
        1-2*(i*i+j*j)),1).view(-1,3,3)

class DenoisingTrainer:
    def __init__(self, meta, model, dset):
        self.meta = meta
        self.model = model.train().cuda()

        self.dset = dset
        self.dloader = DataLoader(dset, shuffle=True, batch_size=self.meta['batchSize'], num_workers=0, drop_last=False)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=2e-4)
        # self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, .9999) # 1000 iters, 90% lr
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, .99)
        self.lossDict = LossDict()
        self.ii = self.meta.get('ii',0)
        self.viz = None

        self.title = self.meta['title']


    def save(self, outDir='/data/human/saves/'):
        f = os.path.join(outDir,'{}.{}.pt'.format(self.title,self.ii))
        torch.save(dict(
            ii=self.ii,
            title=self.title,
            meta=self.meta,
            sd=self.model.state_dict()), f)
        print(' - saved', f)

    def t_to_sigma(self, ts):
        sigma_min = .02
        # sigma_max = 500
        sigma_max = 1
        base = (sigma_max / sigma_min)
        base = torch.FloatTensor((base,)).to(ts.device)
        return sigma_min * base.pow(ts)


    def get_batch(self, loaderIter):
        with torch.no_grad():
            try:
                x = next(loaderIter)
            except StopIteration:
                return None

            B,S = x.shape
            x = x.cuda()
            D = x.device

            ts = torch.rand(B, 1, device=D)
            sigs = self.t_to_sigma(ts)
            rs = torch.randn(B, S, device=D) * sigs

            nx = x + rs
            # nx = encodeTime(nx,ts)

            if self.model.conditional:
                x,nx,ts,z,cams = self.get_camData_for_batch(x,nx,ts)
            else:
                z,cams = None

        return x, nx, ts, z, cams

    # Requires batch size of next(loaderIter) to be 1!
    def get_batch_interpTime(self, loaderIter, steps=32):
        with torch.no_grad():
            try:
                x = next(loaderIter)
            except StopIteration:
                return None

            B,S = x.shape
            assert B == 1

            x = x.cuda()
            D = x.device

            #
            # FIXME: Also try lying to the network about the noise -- to see what effect it has.
            # I'd probably want a side-by-side viz thouhg.
            #
            N = steps
            ts = 1-torch.linspace(0,1,N,device=D)
            # print(ts.shape)
            sigs = self.t_to_sigma(ts).view(N,1)
            # print(sigs.shape)
            rs = torch.randn(1, S, device=D) * sigs # NOTE: replicate same sample across all N!
            # print(rs.shape)

            nx = x + rs
            # print(nx.shape)

            # nx = encodeTime(nx,ts)
            x = x.repeat(N,1)

            if self.model.conditional:
                x,nx,ts,z,cams = self.get_camData_for_batch(x[:1],nx,ts)
                x = x.repeat(N,1)
            else:
                z,cams = None,None

            z = z.repeat(N,1)

        return x, nx, ts, z, cams

    def get_camData_for_batch(self, x, nx, ts):
        (N0,S),d = x.size(), x.device

        def try_gen(x):
            (N,S) = x.size()
            eye = (torch.rand(N,1,3,device=d)-.5) * torch.FloatTensor((1.4,1.4,.5)).view(1,1,3).to(d) + torch.FloatTensor((0,1,-2)).view(1,1,3).to(d)
            q = (torch.rand(N,4,device=d)-.5) * torch.FloatTensor((1,0,1,0)).view(1,4).to(d) + torch.FloatTensor((8,0,0,0)).to(d)
            R = q_to_matrix(F.normalize(q,dim=1))
            wh = torch.rand(N,2,device=d) * 0 + 512
            uv = torch.rand(N,2,device=d) * 0 + 1
            # now we have points and tensors analogous to MVP matrices, and can project.
            x = x.view(N,-1,3) - eye
            x = x.view(N,-1,3) @ R.permute(0,2,1) # same size as x
            x = x[...,:2] / x[...,2:]
            x[...,:2] = x[...,:2] * (wh/uv).view(N,1,2) + wh.view(N,1,2)*.5
            cams = torch.cat((uv,wh,eye.flatten(1),R.flatten(1)), 1)
            return x,cams

        y,cams = try_gen(x)

        bad = ((y < 0) | (y > 512)).view(N0,-1).any(1)
        trials = 0
        maxTrials = 33 if N0 > 1 else 99999
        while (bad==True).any():
            # print(trials,bad.float().mean())
            xx = x[bad]
            yy,ccs = try_gen(xx)
            y[bad] = yy
            cams[bad] = ccs
            bad = ((y < 0) | (y > 512)).view(N0,-1).any(1)
            trials += 1
            if trials > maxTrials:
                # print(f' - Warning: failed to sample good xform for {bad.sum().cpu().item()} examples after {maxTrials} trials')
                # assert False, 'too many trials'
                good = ~bad
                return x[good], nx[good], ts[good], y[good].view(-1,y.size(1)*y.size(2)), cams[good]

        return x, nx, ts, y.view(N0,-1), cams




    def train_batch(self, batch):
        x,nx,ts,z,cams = batch
        B,S = x.size()


        self.model = self.model.train()
        self.opt.zero_grad()
        s = self.model(nx,ts,z)

        loss = (s - (x-nx[...,:S])).pow(2).sum(-1).mean()
        loss.backward()

        self.opt.step()

        self.lossDict.push(self.ii, 'dn', loss)
        if self.ii % 250 == 0:
            self.lossDict.print(self.ii,lr=self.scheduler.get_last_lr()[0])

        self.ii += 1

    def show_viz(self):
        from .renderExamples import ExampleRenderer
        if self.viz is None:
            self.viz = ExampleRenderer(1024,1024)
            self.viz.init(True)

        with torch.no_grad():

            # g = torch.Generator(0)
            g = None
            loader = DataLoader(self.dset, shuffle=True, batch_size=1, num_workers=0, generator=g, drop_last=False)
            loader_iter = iter(loader)
            self.model = self.model.eval()

            if 0:
                self.model = self.model.eval()
                def genDataDicts():
                    while (batch := self.get_batch(loader_iter)) is not None:
                        x,nx,ts,z,cams = batch
                        score = self.model(nx,ts,z)
                        yield dict(x=x.cpu().numpy(),nx=nx.cpu().numpy(),ts=ts.cpu().numpy(),estScore=score.cpu().numpy(), inds=self.dset.inds, z=z,cams=cams)
                self.viz.runBasicUntilQuit(genDataDicts)

            else:
                def genDataDicts():
                    while (batch := self.get_batch_interpTime(loader_iter)) is not None:
                        x,nx,ts,z,cams = batch
                        score = self.model(nx,ts,z)
                        yield dict(true=x.cpu().numpy(),noisy=nx.cpu().numpy(),ts=ts.cpu().numpy(),estScore=score.cpu().numpy(), inds=self.dset.inds, z=z,cams=cams)
                self.viz.runAnimatedUntilQuit(genDataDicts)

            self.model = self.model.train()




    def train(self):
        stop = False
        for epoch in range(self.meta['epochs']):
            print(f' - Epoch {epoch:>3d}')
            self.dloader_iter = iter(self.dloader)

            try:
                while (batch := self.get_batch(self.dloader_iter)) is not None:
                    self.train_batch(batch)
            except KeyboardInterrupt:
                while 1:
                    print("\n - enter 'v' to viz, 'q' to stop, 's' to save")
                    s = input()
                    if s.startswith('v'):
                        self.show_viz()
                        break
                    if s.startswith('q'):
                        stop = True
                        break
                    if s.startswith('s'):
                        self.save()
                        break
            if stop: break
            self.scheduler.step()



if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dsetFile', default='/data/human/posePrior/procStride10/data.npz')
    parser.add_argument('--modelNo', default=0)
    parser.add_argument('--batchSize', default=256, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--load', default=None)
    parser.add_argument('--kind', default='resnet2')
    parser.add_argument('--title', default='firstModel1')
    args = parser.parse_args()

    meta = dict(args._get_kwargs())

    dset = PosePriorDataset(args.dsetFile, masterScale=.03)
    meta['S'] = dset.getStateSize()

    # Setup conditional data.
    meta['Z'] = (meta['S'] * 2) // 3
    meta['Cz'] = 2

    meta['kind'] = args.kind
    meta['timeEncoderKind'] = 'identSquareCube'
    meta['load'] = args.load
    meta['iter'] = meta['epoch'] = 0
    meta['title'] = args.title

    model,meta = get_model(dset, meta)
    print(model)

    t = DenoisingTrainer(meta, model, dset)
    t.train()
