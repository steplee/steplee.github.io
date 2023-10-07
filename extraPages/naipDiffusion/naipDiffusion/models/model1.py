import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.ops import Conv2dNormActivation

class ResidualBlock(nn.Module):
    def __init__(self, A, B, cna, down=True):
        super().__init__()
        k
        # hidden = B * 3 / /2
        hidden = B

        self.f = nn.Sequential(
                cna(A, hidden),
                cna(hidden, A))
        self.down = nn.AvgPool2d(2,2) if down else None
        self.out = cna(A, B)

    def forward(self, x):
        y = self.f(x)
        z = y + x
        if self.down is not None: z = self.down(z)
        return self.out(Z)

class FlatPositionEncoder(nn.Module):
    def __init__(self, C):
        super().__init__()
        # self.f = nn.Linear(1, C, bias=True)
        self.f = nn.Sequential(
                nn.Linear(1, C//2, bias=True),
                nn.ReLU(True),
                nn.Linear(C//2, C, bias=True))
    def forward(self, b, h, w):
        t = torch.linspace(-1,1,steps=h*w,device=self.f.weight.device)
        return self.f(t).view(1,h,w,C).permute(0,3,1,2).repeat(B,1,1,1)
class TimeEncoder(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.f = nn.Sequential(
                nn.Linear(1, C//2, bias=True),
                nn.ReLU(True),
                nn.Linear(C//2, C, bias=True))
    def forward(self, t, h, w):
        B = t.size(0)
        return self.f(t).view(B,1,1,C).repeat(1,h,w,1)


class Model_1(BaseModel):
    def __init__(self, meta):
        super().__init__(meta)

        act = lambda: nn.ReLU(True)
        nl, bias = nn.BatchNorm2d, False

        def cna(inC, outC, K, stride=1, groups=1):
            return Conv2dNormActivation(
                    inC, outC, K,
                    stride=stride,
                    groups=groups,
                    padding=K//2,
                    norm_layer=nl,
                    activation_layer=act,
                    bias=bias)


        self.top = nn.Sequential(
                cna(3,32,stride=2),
                res_block(64, 128),
                res_block(128, 256),
                res_block(256, 512),
                res_block(512, 1024))

        # cat
        # vizFeatDims = 1024
        # timeFeatDims = 32
        # xformFeatDims = vizFeatDims + timeFeatDims

        # sum
        featDims = 1024
        xformFeatDims = featDims

        self.xformer = MyTransformer(xformFeatDims)

    def _forward(self, x, t):
        B,C,H,W = x.size()


        f = self.top(x)
        _,_,FH,FW = f.size()

        et = self.encodeTime(t, h=FH, w=FW)
        ep = self.encodePosition(B,H,W)
        f = f + et + ep

        f = self.xformer(f)

        y = self.decode(f)
        return y

