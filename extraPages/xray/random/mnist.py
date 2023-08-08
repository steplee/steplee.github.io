import torch, torch.nn as nn, torch.nn.functional as F
import cv2, numpy as np

device = torch.device('cuda')

class SimpleAe(torch.nn.Module):
    def __init__(self, H):
        super().__init__()

        # self.f = nn.Sequential(nn.Linear(28*28, H), nn.ReLU(True), nn.Linear(H,H))
        self.f = nn.Sequential(nn.Linear(28*28, H))
        self.g = nn.Linear(H, 28*28)
        # self.g = nn.Sequential(nn.Linear(H, 768), nn.ReLU(True), nn.Linear(768,784))

    def forward(self, x):
        B,H,W = x.size()
        x = x.flatten(1)
        x = (x.float() - 128) / 128
        z = self.f(x)
        px = self.g(z).view(B,H,W)
        return None, z, px

    def loss(self, x, zz, z, px):
        return (x - px).abs().mean()

class Vae(torch.nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H

        self.mode = 'bernoulli'
        # self.mode = 'gauss'

        # self.f = nn.Sequential(nn.Linear(28*28, H), nn.ReLU(True), nn.Linear(H,H))
        self.f = nn.Sequential(nn.Linear(28*28, 2*H))
        if self.mode == 'bernoulli':
            self.g = nn.Sequential(nn.Linear(H, 28*28), nn.Sigmoid())
        else:
            self.g = nn.Linear(H, 28*28)
        # self.g = nn.Sequential(nn.Linear(H, 768), nn.ReLU(True), nn.Linear(768,784))


    def forward(self, x):
        B,H,W = x.size()
        x = x.flatten(1)
        x = (x.float() - 128) / 128

        zz = self.f(x).view(B, self.H, 2)
        mu,sig = zz[...,0], zz[...,1]
        sig = F.softplus(sig)
        z = mu + torch.randn_like(sig) * sig

        px = self.g(z).view(B,H,W)
        return zz, z, px

    def loss(self, x, zz, z, px):
        if self.mode == 'bernoulli':
            recons = (x*px.log() + (1-x)*(1-px).log()).mean()
        else:
            # I guess for gaussian, you need mean and cov.
            # assert False
            recons = -(x-px).pow(2).mean() * 1e-1

        mu,sig = zz[...,0], zz[..., 1]
        likeli = (1 + sig.pow(2).log() - mu.pow(2) - sig.pow(2)).mean()
        print('recons',recons.item())
        print('likeli',recons.item())

        return -(recons + likeli)





def train_simple(model):

    x,label = torch.load('/data/mnist/raw/processed/training.pt')
    dataset = torch.utils.data.TensorDataset(x)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    vizBatch = next(iter(dataloader))[0].to(device)[:64]
    def showViz():
        with torch.no_grad():
            x = vizBatch
            B,H,W = x.size()

            dimg = np.zeros((B*H, W*2), dtype=np.uint8)
            dimg[:, :W] = x.view(-1,W).cpu().numpy()

            zz,z,px = model.eval()(x)
            px = (px * 128 + 128).clamp(0,255).byte().cpu().numpy()
            dimg[:, W:] = px.reshape(-1,W)

            BB = B // 8
            dimg1 = np.zeros((BB*H, 2*W*8), dtype=np.uint8)
            for x in range(8):
                dimg1[:, x*2*W:(x+1)*2*W] = dimg[x*H*BB:(x+1)*H*BB]

            cv2.imshow('gen',dimg1)
            cv2.waitKey(0)

    for epoch in range(30):
        for ii,batch in enumerate(dataloader):
            x = batch[0].to(device)
            zz,z,px = model.train().forward(x)
            loss = model.loss(x,zz,z,px)
            # print(loss.item())
            loss.backward()
            opt.step()
            opt.zero_grad()

        print('epoch',epoch)
        showViz()


# model = SimpleAe(H=256).train().to(device)
# model = Vae(H=64).train().to(device)
model = Vae(H=128).train().to(device)
# model = Vae(H=1024).train().to(device)
train_simple(model)
