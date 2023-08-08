import torch, torch.nn.functional as F, numpy as np
import matplotlib.pyplot as plt

def noprint(*a): pass
def get_lap_analytic(doPrint=False):
    from sympy import symbols, exp, sin, diff, simplify, sqrt, pprint, lambdify
    x,y,z = symbols('x y z')
    f = exp(-.5*sqrt(x*x + y*y + z*z))
    d2f_x2 = (diff(f, x, x))
    d2f_y2 = (diff(f, y, y))
    d2f_z2 = (diff(f, z, z))
    lap_f = (d2f_x2 * d2f_y2 * d2f_z2)
    lap_f_ = lap_f.subs(x*x+y*y+z*z, 'r2')
    if doPrint:
        print('\ndf_dx:')
        pprint(diff(f,x).subs(x*x+y*y+z*z, 'r2'))
        print('\nd2f_dx2:')
        pprint(d2f_x2.subs(x*x+y*y+z*z, 'r2'))
        print(' - These would be the same for y/z, just replace the \'x\' with them.')
        print('\nlap:')
        pprint(lap_f_)
    return lambdify((x,y,z), lap_f)
# get_lap_analytic(doPrint=True)

def eval_gauss_lap(p):
    r2 = (p*p).sum(-1)
    r  = r2.sqrt()
    a = .25 * p*p / r2
    b = -.5 / r
    b = b.view(-1,1)
    c = .5 * p*p / ((r2) ** (1.5))
    return (a+b+c).prod(-1) * (-1.5*r).exp()
lf=get_lap_analytic()
print(lf(0,1,0), eval_gauss_lap(torch.FloatTensor([0,1,0]).view(1,3)))
print(lf(-1.5799,.0223,0), eval_gauss_lap(torch.FloatTensor([-1.5799,.0223,0]).view(1,3)))


def match_finite_diff_to_closed_form():
    lap_f = get_lap_analytic()
    # STOPPED HERE. Need to draw diagrams to see how scale works.


def viz_conv_of_lap():
    import cv2

    if 1:
        hw = 24
        img = torch.zeros((hw,hw), dtype=torch.float32).cuda()
        img[hw//2, hw//2] = 1
        imgs = [img]
        K = torch.ones((1,1,3,3)).cuda()
        # K[:,:,1,1] *= 8
        K /= K.sum()
        for i in range(3):
            # img = F.avg_pool2d(img.unsqueeze(0).unsqueeze(0), 3, 1, padding=1)
            img = F.conv2d(img.unsqueeze(0).unsqueeze(0), K, padding=1)
            img = img[0,0]
            imgs.append(img)
        imgs = (torch.stack(imgs, 0))

    baseImg = imgs[1]
    baseImg = imgs[2]

    if 1:
        print('\nDoing 2d divergence and laplacian of regular blur.')

        L = torch.cuda.FloatTensor((0,1,0,1,-4,1,0,1,0)).reshape(1,1,3,3)
        # L = torch.cuda.FloatTensor((1,1,1,1,-8,1,1,1,1)).reshape(1,1,3,3)
        Dy = torch.cuda.FloatTensor((-1,0,1)).view(1,1,3,1)
        Dx = torch.cuda.FloatTensor((-1,0,1)).view(1,1,1,3)
        img = baseImg.unsqueeze(0).unsqueeze(0)
        dy,dx = F.conv2d(img, Dy, padding=1)[...,1:-1], F.conv2d(img, Dx, padding=1)[...,1:-1,:]
        ddy,ddx = F.conv2d(dy, Dy, padding=1)[...,1:-1], F.conv2d(dx, Dx, padding=1)[...,1:-1,:]

        lap = F.conv2d(img, L, padding=1)[...,1:-1]
        # lap = (ddx + ddy)

        P = img.size(2)//2
        print(lap.shape,img.shape,P)
        convLap = F.conv2d(lap, img, padding=P)[...,1:-1,1:-1]
        print(convLap.shape)

        gradDotProd  = -F.conv2d(dy, dy, padding=P)[...,1:-1,1:-1]
        gradDotProd += -F.conv2d(dx, dx, padding=P)[...,1:-1,1:-1]


        for name, arr in zip(('baseImg', 'lap', 'F * lap', 'deltaF*deltaF'), (img, lap, convLap, gradDotProd)):
            arr = arr.div(abs(arr.max()))
            arr = arr[0,0]
            print(f' - {name} nnz = {(arr!=0).sum()}')
            arr = torch.stack((
                (-arr).clamp(0,1000),
                torch.zeros_like(arr),
                ( arr).clamp(0,1000)), -1) # NHW -> NHW3

            arr = (arr.cpu().numpy() * 255).astype(np.uint8)
            # arr = np.hstack(arr)
            arr = cv2.resize(arr, (0,0), fx=16, fy=16, interpolation=cv2.INTER_NEAREST)
            cv2.imshow(name, arr)
    # cv2.waitKey(0)
    # exit()
