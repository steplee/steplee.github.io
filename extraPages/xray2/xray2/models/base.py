import torch, torch.nn as nn, torch.nn.functional as F, os

# --------------------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------------------

class NoiseLevelToSigma_Exp(nn.Module):
    def __init__(self, f, a, bias):
        super().__init__()
        self.f, self.a = f, a
        self.bias=bias
    def forward(self, nl):
        sig = torch.exp(nl * self.a + self.bias) * self.f
        return sig

class NoiseLevelToSigma_Linear(nn.Module):
    def __init__(self, max, min):
        super().__init__()
        self.max, self.min = max,min
    def forward(self, nl):
        # Okay: I lied about the linear part.
        # sig = (nl**3)*(self.max-self.min) + self.min
        # sig = (nl**2)*(self.max-self.min) + self.min
        sig = (nl**3)*(self.max-self.min) + self.min
        return sig

class NoiseLevelEncoder(nn.Module):
    def __init__(self, meta):
        super().__init__()
        powers = meta['nlPowers']
        powers = torch.FloatTensor([powers])
        self.register_buffer('powers', powers)
    def getSize(self): return self.powers.numel()
    def forward(self, nl):
        return self.powers.view(1,-1) * nl - .5

# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------

class BaseModel(nn.Module):
    def __init__(self, meta):
        super().__init__()
        meta.setdefault('nlMapKind', 'exp')
        meta.setdefault('nlMapF', .01)
        meta.setdefault('nlMapA', 5.0)
        meta.setdefault('nlMapBias', .0)
        meta.setdefault('nlMapMin', .02)
        meta.setdefault('nlMapMax', 1.4)
        self.meta = meta
        # for k,v in meta.items(): setattr(self, k, v)
        self.title = meta['title']
        self.stateSize = meta['stateSize']
        self.conditionalSize = meta['conditionalSize']
        self.inputSize = meta['inputSize']

        if meta['nlMapKind'] == 'exp':
            self.noiseLevelToSigma = NoiseLevelToSigma_Exp(meta['nlMapF'], meta['nlMapA'], meta['nlMapBias'])
        else:
            self.noiseLevelToSigma = NoiseLevelToSigma_Linear(meta['nlMapMax'], meta['nlMapMin'])
        self.noiseLevelEncoder = NoiseLevelEncoder(meta)

    def mapNoiseLevelToSigma(self, nl):
        return self.noiseLevelToSigma(nl)

    def save(self, iter, dir='/data/human/saves/train2'):
        try: os.makedirs(dir)
        except: pass
        path = os.path.join(dir, f'{self.title}.{iter}.pt')
        torch.save(dict(
            sd=self.state_dict(),
            meta=self.meta,
            clazz=self.__class__.__name__,
            iter=iter), path)
        print(' - Saved', path)

    @staticmethod
    def load(path):
        d = torch.load(path)
        m = globals()[d['clazz']](d['meta'])
        m.load_state_dict(d['sd'])
        return dict(d=d, model=m, iter=d['iter'])

    def loss(self, x, px, nl):
        if self.meta['divideErrorByNl']:
            # e = (abs(x-px).sum(1) / nl.view(-1)).mean()
            # e = (abs(x-px).sum(1) / (1+20*nl).view(-1)).mean()
            e = (abs(x-px).sum(1) / (1+2*nl).view(-1)).mean()
            return e
        else:
            return F.l1_loss(x,px)


class SimpleModel(BaseModel):
    def __init__(self, meta):
        super().__init__(meta)

        norm = nn.BatchNorm1d
        bias = False

        inputSizeWithEncNoise = self.inputSize + self.noiseLevelEncoder.getSize()
        last = inputSizeWithEncNoise

        def mk_blk(c, bias=bias, useNorm=True, useRelu=True):
            nonlocal last
            layers = []
            layers.append(nn.Linear(last,c,bias=bias))
            if useNorm: layers.append(norm(c))
            if useRelu: layers.append(nn.ReLU(True))
            o = nn.Sequential(*layers)
            last = c
            return o

        if 0:
            self.net = nn.Sequential(
                    mk_blk(128),
                    mk_blk(128),
                    mk_blk(256),
                    mk_blk(512),
                    mk_blk(256),
                    mk_blk(128),
                    mk_blk(self.stateSize,bias=True,useNorm=False,useRelu=False))
        else:
            self.net = nn.Sequential(
                    mk_blk(256),
                    mk_blk(1024),
                    mk_blk(2048),
                    mk_blk(1024),
                    mk_blk(2048),
                    mk_blk(512),
                    mk_blk(self.stateSize,bias=True,useNorm=False,useRelu=False))

    def forward(self, y, n, z):
        yy = torch.cat((y, self.noiseLevelEncoder(n), z), 1)
        pdx = self.net(yy)
        return pdx

class ResidualBlock(nn.Module):
    def __init__(self, cin, cmid, cout, norm, bias, finalNorm=True, finalRelu=True):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(cin, cmid, bias=bias),
                norm(cmid),
                nn.ReLU(True),

                nn.Linear(cmid, cmid, bias=bias), norm(cmid), nn.ReLU(True), # NOTE: Bigger mdoel.

                nn.Linear(cmid, cin, bias=bias),
                norm(cin),
                nn.ReLU(True),
                nn.Dropout(p=.3),
            )
        self.out = nn.Sequential(
                nn.Linear(cin, cout, bias=bias),
                norm(cout),
                nn.ReLU(True))

    def forward(self, a):
        b = self.net(a)
        return self.out(b + a)

class ResidualModel(BaseModel):
    def __init__(self, meta):
        super().__init__(meta)
        norm = nn.BatchNorm1d
        bias = False

        inputSizeWithEncNoise = self.inputSize + self.noiseLevelEncoder.getSize()
        last = inputSizeWithEncNoise

        def mk_blk(cmid, cout):
            nonlocal last
            o = ResidualBlock(last, cmid, cout, norm, bias)
            last = cout
            return o

        self.net = nn.Sequential(
                mk_blk(128, 256),
                mk_blk(1024, 512),
                mk_blk(2048, 1024),

                # mk_blk(1024, 512),
                # mk_blk(512, 256),
                mk_blk(2048, 512),
                mk_blk(2048, 256),

                nn.Sequential(nn.Linear(last,self.stateSize,bias=True)))

    def forward(self, y, n, z):
        yy = torch.cat((y, self.noiseLevelEncoder(n), z), 1)
        pdx = self.net(yy)
        return pdx

class ResidualModel2(BaseModel):
    def __init__(self, meta):
        super().__init__(meta)
        # norm = nn.BatchNorm1d
        norm = nn.InstanceNorm1d
        bias = False
        inputSizeWithEncNoise = self.inputSize + self.noiseLevelEncoder.getSize()
        last = inputSizeWithEncNoise

        def mk_blk(cmid, cout):
            nonlocal last
            o = ResidualBlock(last, cmid, cout, norm, bias)
            last = cout
            return o

        self.net = nn.Sequential(
                mk_blk(128, 256),
                mk_blk(1024, 512),
                mk_blk(2048, 1024),
                mk_blk(2048, 1024),
                nn.Sequential(nn.Linear(last,self.stateSize,bias=True)))

    def forward(self, y, n, z):
        z = z - .5
        yy = torch.cat((y, self.noiseLevelEncoder(n), z), 1)
        pdx = self.net(yy)
        return pdx



def get_model(meta):
    if 'load' in meta and len(meta['load']) > 0:
        return BaseModel.load(meta['load'])

    if meta['modelKind'] == 'simple':
        return dict(model=SimpleModel(meta), iter=0)
    if meta['modelKind'] == 'residual':
        return dict(model=ResidualModel(meta), iter=0)
    if meta['modelKind'] == 'residual2':
        return dict(model=ResidualModel2(meta), iter=0)


