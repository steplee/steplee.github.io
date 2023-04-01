import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class TimeEncoder_Simple(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x,t):
        with torch.no_grad():
            B,S = x.size()
            et = t.view(B,1)
            return torch.cat((x,et), 1)
    def getSize(self, S): return S+1
class TimeEncoder_IdentSquareCube(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x,t):
        with torch.no_grad():
            B,S = x.size()
            et = t.view(B,1)
            e = et, et**2, et**3
            return torch.cat((x,*e), 1)
    def getSize(self, S): return S+3

def get_time_encoder(meta):
    kind = meta['timeEncoderKind']
    if kind == 'simple':
        return TimeEncoder_Simple()
    if kind == 'identSquareCube':
        return TimeEncoder_IdentSquareCube()
    assert False

class ResidualNet1(nn.Module):
    def __init__(self, meta):
        super().__init__()

        S = meta['S']
        assert meta['Z'] == 0, 'ResNet1 does not support conditional mode'
        self.conditional = False
        self.timeEncoder = get_time_encoder(meta)
        cin = self.timeEncoder.getSize(S)

        bias = False
        self.mod1 = nn.Sequential(
                nn.Linear(cin, 256, bias=bias),
                nn.BatchNorm1d(256))
        self.mod2 = nn.Sequential(
                nn.ReLU(True),
                nn.Linear(256, 512, bias=bias),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Linear(512, 512, bias=bias),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Linear(512, 256, bias=bias),
                nn.BatchNorm1d(256),
                )
        self.mod3 = nn.Sequential(
                nn.ReLU(True),
                nn.Linear(256, 256, bias=bias),
                nn.BatchNorm1d(256))
        self.fin = nn.Sequential(
                nn.ReLU(True),
                nn.Linear(256, S))


    def forward(self, s, t, z):
        a = self.timeEncoder(s,t)

        b = self.mod1(a)
        c = self.mod2(b) + b
        d = self.mod3(c) + c
        e = self.fin(d)
        return e

# Like above (fully connected), but is also conditional (again: fully-connected, simplest arch possible)
class ResidualNet2(nn.Module):
    def __init__(self, meta):
        super().__init__()

        S = meta['S']
        Z = meta['Z']
        self.timeEncoder = get_time_encoder(meta)
        cin = self.timeEncoder.getSize(S)

        bias = False
        self.mod1 = nn.Sequential(
                nn.Linear(cin, 256, bias=bias),
                nn.BatchNorm1d(256))
        self.conditional_mod1 = nn.Sequential(
                nn.Linear(Z, 256, bias=bias),
                nn.BatchNorm1d(256))
        self.mod2 = nn.Sequential(
                nn.ReLU(True),
                nn.Linear(256, 512, bias=bias),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Linear(512, 512, bias=bias),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Linear(512, 256, bias=bias),
                nn.BatchNorm1d(256),
                )
        self.mod3 = nn.Sequential(
                nn.ReLU(True),
                nn.Linear(256, 512, bias=bias),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Linear(512, 256, bias=bias),
                nn.BatchNorm1d(256),
                )
        self.fin = nn.Sequential(
                nn.ReLU(True),
                nn.Linear(256, S))

        self.conditional = True

    def forward(self, s, t, z):
        # print(z.shape, self.conditional_mod1[0].weight.shape)
        a = self.timeEncoder(s,t)
        b = self.mod1(a) + self.conditional_mod1(z)
        c = self.mod2(b) + b
        d = self.mod3(c) + c
        e = self.fin(d)
        return e

# Not currently working.
class AttnBlock1(nn.Module):
    def __init__(self, cin, cout, numHeads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(cin, numHeads)
        self.net = nn.Sequential(
                nn.Conv1d(cin, cout, 1, bias=False),
                nn.BatchNorm1d(cout),
                # nn.ReLU(True),
                )
        self.relu = nn.ReLU(True)
        self.through = nn.Sequential(
            nn.Conv1d(cin, cout, 1, bias=False),
            nn.BatchNorm1d(cout),
        )
        self.w = None
        self.keepWeight = True

    def forward(self, a):
        a1 = a.permute(2,0,1)
        b,w = self.attn(a1,a1,a1)
        b = b.permute(1,2,0)
        c = self.net(b)
        d = self.relu(self.through(a) + c)
        if self.keepWeight: self.w = w
        return d

class AttnNet1(nn.Module):
    # def __init__(self, Cin, C, S, CN, Z=0):
    def __init__(self, meta):
        super().__init__()

        S = meta['S']
        self.L = S // 3
        self.timeEncoder = get_time_encoder(meta)
        cin = self.timeEncoder.getSize(S)
        ct = cin - S

        Z = meta['Z']
        Cz = meta['Cz'] # size of each subvector
        Nz = Z // Cz # num sub vectors
        self.conditional = True
        # assert Nz == self.L, 'AttnNet1 only works when Nz == L (conditional subvectors = state subvectors)'
        self.Nz, self.Cz = self.Nz, self.Cz

        Cposition = 1
        # cinEach = 3 + ct + Cposition + Z//self.L
        cinEach = 3 + ct + Cposition
        selfcinEach = cinEach
        # print(cinEach,'=',3,ct,Cposition,Z//S)

        self.pre = nn.Sequential(
                nn.Conv1d(cinEach, 256, 1, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(True))

        self.mods = nn.ModuleList([
                AttnBlock1(256, 512),
                AttnBlock1(512,512),
                AttnBlock1(512,256)])

        self.fin = nn.Sequential(
                nn.Conv1d(256, 3, 1))


    def forward(self, s, t, z):
        (B,S),L,Lz = s.size(), self.L,self.Nz


        # position encoding is just a number from [-1,1].
        # It's not really position here, but helps to label what joint an item is.
        pos = torch.linspace(-1,1, L, device=s.device).view(1,-1,1)


        at = self.timeEncoder(s,t)
        ST = at.size(1)
        aa,tt = at[:,:S], at[:,S:]
        a = torch.cat((
            aa.view(B,L,3),
            tt.view(B,1,-1).repeat(1,L,1),
            pos.view(1,L,-1).repeat(B,1,1),
            # z.view(B,L,-1)
        ), -1) \
            .permute(0,2,1) # NL(S+T+P+Z) => BCL
            #.permute(1,0,2) # NL(S+T+P+Z) => LN(S+T+P+Z)
        # print(a.size())

        # Add conditioning information.
        zpad = self.Cz - self.cinEach
        zpad = torch.zeros((1,1,zpad),device=s.device).repeat(B,Lz,1)
        a = torch.cat((
            z.view(B,Lz,-1),
            zpad), -1).permute(0,2,1)

        a = self.pre(a)
        for mod in self.mods:
            a = mod(a)
        e = self.fin(a)
        e = e.view(B,S)
        return e

class SplitAttnNet1(nn.Module):
    def __init__(self, meta):
        super().__init__()

        S = meta['S']
        self.L = S // 3
        self.timeEncoder = get_time_encoder(meta)
        cin = self.timeEncoder.getSize(S)
        ct = cin - S

        Z = meta['Z'] # full size
        Cz = meta['Cz'] # size of each subvector
        Lz = Z // Cz # num sub vectors
        self.conditional = True
        self.Z,self.Lz,self.Cz = Z,Lz,Cz

        Cposition = 1
        cinEach = 3 + ct + Cposition
        # print(cinEach,'=',3,ct,Cposition,Z//S)

        self.pre = nn.Sequential(
                nn.Conv1d(cinEach, 256, 1, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(True))

        cinEach_z = Cz + Cposition
        self.pre_z = nn.Sequential(
                nn.Conv1d(cinEach_z, 256, 1, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(True))

        self.mods = nn.ModuleList([
                AttnBlock1(256, 512),
                AttnBlock1(512,512),
                AttnBlock1(512,256)])

        self.fin = nn.Sequential(
                nn.Conv1d(256, 3, 1))


    def forward(self, s, t, z):
        (B,S),L,Lz = s.size(), self.L, self.Lz

        # position encoding is just a number from [-1,1].
        # It's not really position here, but helps to label what joint an item is.
        pos   = torch.linspace(-1,1, L, device=s.device).view(1,-1,1)
        pos_z = torch.linspace(-1,1, Lz, device=s.device).view(1,-1,1)

        at = self.timeEncoder(s,t)
        ST = at.size(1)

        Z,Lz,Cz = self.Z,self.Lz,self.Cz
        aa,tt = at[:,:S], at[:,S:]
        a = torch.cat((
            aa.view(B,L,3),
            tt.view(B,1,-1).repeat(1,L,1),
            pos.view(1,L,-1).repeat(B,1,1),
        ), -1) \
            .permute(0,2,1) # NL(S+T+P+Z) => BCL

        z = torch.cat((
            z.view(B,Lz,Cz),
            pos_z.view(1,Lz,-1).repeat(B,1,1),
        ), -1) \
            .permute(0,2,1) # NLC => BCL

        a = self.pre(a)
        z = self.pre_z(z)
        a = torch.cat((a,z), -1) # Make sequence length L+Lz.
        for mod in self.mods:
            a = mod(a)
        a = a[:,:,:L] # Remove conditional data.
        e = self.fin(a)
        e = e.view(B,S)
        return e

    def get_attn_maps(self):
        maps = []
        for mod in self.mods:
            maps.append(mod.w)
        return maps

class SplitAttnNet2(nn.Module):
    def __init__(self, meta):
        super().__init__()

        S = meta['S']
        self.L = S // 3
        self.timeEncoder = get_time_encoder(meta)
        cin = self.timeEncoder.getSize(S)
        ct = cin - S

        Z = meta['Z'] # full size
        Cz = meta['Cz'] # size of each subvector
        Lz = Z // Cz # num sub vectors
        self.conditional = True
        self.Z,self.Lz,self.Cz = Z,Lz,Cz

        Cposition = 1
        cinEach = 3 + ct + Cposition

        self.pre = nn.Sequential(
                nn.Conv1d(cinEach, 256, 1, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(True))

        cinEach_z = Cz + Cposition
        self.pre_z = nn.Sequential(
                nn.Conv1d(cinEach_z, 256, 1, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(True))

        self.mods = nn.ModuleList([
                AttnBlock1(256, 512),
                AttnBlock1(512,512),
                AttnBlock1(512,1024),
                AttnBlock1(1024,2048),
                AttnBlock1(2048,512)])

        self.fin = nn.Sequential(
                nn.Conv1d(512, 3, 1))


    def forward(self, s, t, z):
        (B,S),L,Lz = s.size(), self.L, self.Lz

        # position encoding is just a number from [-1,1].
        # It's not really position here, but helps to label what joint an item is.
        pos   = torch.linspace(-1,1, L, device=s.device).view(1,-1,1)
        pos_z = torch.linspace(-1,1, Lz, device=s.device).view(1,-1,1)

        at = self.timeEncoder(s,t)
        ST = at.size(1)

        Z,Lz,Cz = self.Z,self.Lz,self.Cz
        aa,tt = at[:,:S], at[:,S:]
        a = torch.cat((
            aa.view(B,L,3),
            tt.view(B,1,-1).repeat(1,L,1),
            pos.view(1,L,-1).repeat(B,1,1),
        ), -1) \
            .permute(0,2,1) # NL(S+T+P+Z) => BCL

        z = torch.cat((
            z.view(B,Lz,Cz),
            pos_z.view(1,Lz,-1).repeat(B,1,1),
        ), -1) \
            .permute(0,2,1) # NLC => BCL

        a = self.pre(a)
        z = self.pre_z(z)
        a = torch.cat((a,z), -1) # Make sequence length L+Lz.
        for mod in self.mods:
            a = mod(a)
        a = a[:,:,:L] # Remove conditional data.
        e = self.fin(a)
        e = e.view(B,S)
        return e

    def get_attn_maps(self):
        maps = []
        for mod in self.mods:
            maps.append(mod.w)
        return maps

def get_model(meta0):
    meta = meta0

    loaded_d = None
    if 'load' in meta and meta['load'] != None:
        loaded_d = torch.load(meta['load'])
        meta = loaded_d['meta']
        # TODO: Replace any overwritable paratmers here (lik we want LR to be newly configured, not old)
        if 'batchSize' in meta0: meta['batchSize'] = meta0['batchSize']
        if 'lr' in meta0: meta['lr'] = meta0['lr']
        if 'baseLr' in meta0: meta['baseLr'] = meta0['baseLr']
        if 'epochs' in meta0: meta['epochs'] = meta0['epochs']
        if 'noRandomlyYaw' in meta0: meta['noRandomlyYaw'] = meta0['noRandomlyYaw']
        if 'lossType' in meta0: meta['lossType'] = meta0['lossType']
        if 'ii' in loaded_d: meta['ii'] = loaded_d['ii']

    kind = meta['kind']

    if kind == 'simple':
        S = meta['S']
        inputChannels = S + meta['timeEncoderSize']
        c = inputChannels
        normLayer, bias = nn.BatchNorm1d, False
        def make_block(nc, doNormRelu=True):
            nonlocal c
            oc = c
            c = nc
            if doNormRelu:
                return nn.Linear(oc, nc, bias=bias), normLayer(nc), nn.ReLU(True)
                # return nn.Linear(oc, nc, bias=True), nn.ReLU(True)
            else:
                return (nn.Linear(oc, nc),)
        net = nn.Sequential(
                *make_block(128),
                # *make_block(128),
                # *make_block(256),
                *make_block(128),
                *make_block(128),
                *make_block(S, doNormRelu=False))

    elif kind[:3] == 'res':
        # net = ResidualNet1(inputChannels, S)
        net = ResidualNet2(meta)

    elif kind == 'attn1':
        net = SplitAttnNet1(meta)
    elif kind == 'attn2':
        net = SplitAttnNet2(meta)



    if loaded_d is not None:
        net.load_state_dict(loaded_d['sd'])

    return net, meta

