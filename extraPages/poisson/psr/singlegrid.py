import torch, torch.nn.functional as F
import numpy as np, sys, time
torch.set_printoptions(linewidth=150, edgeitems=10)

from .solvers import solve_cg, solve_cg_AtA
from ..polygonalization.run_marching_algo import run_marching, show_marching_viz

'''

Implementing a 4- or 8-neighbor laplacian as a single convolution is more correct
than doing in two seperate steps dx/dy steps then adding ddx+ddy because that
results in a **dilated** aperature.

Relation between the two PSR papers (original and screened):
    - A_ij is like L_o,o` with the two dilated convs. Depending on impl can be slightly different, but is basically the same.
        -> That is, <lap[A],B> ~=~ <grad[A], grad[B]>
        -> The first is a scalar function inner-product, the second is a function inner-product with vector-inner-product kernel.
        -> The answers are the same (well in discrete implementation, depends on exact filter kernels etc)

'''

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
# viz_conv_of_lap()

# Vizualize the box filter (iterated three times) in 2D.
# The 3d section would suggest that Khazdan 2006 is talking about the regular (non-manhattan) box blur.
# It has roughly twice the number of non-zeros.
def viz_box():
    import cv2

    # Show box blur
    if 1:
        print('\nDoing 2d regular blur.')
        hw = 24
        img = torch.zeros((hw,hw), dtype=torch.float32).cuda()
        img[hw//2, hw//2] = 1
        imgs = [img]
        selfConvs = []
        numberConvolvedNonzeros = [1]
        K = torch.ones((1,1,3,3)).cuda()
        # K[:,:,1,1] *= 8
        K /= K.sum()
        for i in range(3):
            # img = F.avg_pool2d(img.unsqueeze(0).unsqueeze(0), 3, 1, padding=1)
            img = F.conv2d(img.unsqueeze(0).unsqueeze(0), K, padding=1)
            selfConv = F.conv2d(img, img, padding=hw//2)
            numberConvolvedNonzeros.append((selfConv > 0).sum().item())
            img = img[0,0]
            imgs.append(img)
            selfConvs.append(selfConv[0,0])
        imgs = (torch.stack(imgs, 0))
        selfConvs = (torch.stack(selfConvs, 0))

        print(imgs.view(imgs.size(0),-1).sum(1)) # Sum is preserved
        print('nonzeros per iter', (imgs>0).view(imgs.size(0),-1).sum(1))
        print('number non-zero in self-convolutions', numberConvolvedNonzeros)
        # imgs.sqrt_() # Easier to see.

        imgs = (imgs.cpu().numpy() * 255).astype(np.uint8)
        imgs = np.hstack(imgs)
        imgs = cv2.resize(imgs, (0,0), fx=16, fy=16, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('box', imgs)
        selfConvs = (selfConvs.cpu().numpy() * 255).astype(np.uint8)
        selfConvs = np.hstack(selfConvs)
        selfConvs = cv2.resize(selfConvs, (0,0), fx=16, fy=16, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('box (selfConvs)', selfConvs)

    # Show laplacian[box blur]
    if 1:
        print('\nDoing 2d divergence and laplacian of regular blur.')
        hw = 16
        img = torch.zeros((hw,hw), dtype=torch.float32).cuda()
        img[hw//2, hw//2] = 1
        imgs = [img]
        for i in range(3):
            img = F.avg_pool2d(img.unsqueeze(0).unsqueeze(0), 3, 1, padding=1)[0,0]
            imgs.append(img)

        laps = []
        divs = []
        for img in imgs:
            Dy = torch.cuda.FloatTensor((-1,0,1)).view(1,1,3,1)
            Dx = torch.cuda.FloatTensor((-1,0,1)).view(1,1,1,3)
            img = img.unsqueeze(0).unsqueeze(0)
            dy,dx = F.conv2d(img, Dy, padding=1)[...,1:-1], F.conv2d(img, Dx, padding=1)[...,1:-1,:]
            ddy,ddx = F.conv2d(dy, Dy, padding=1)[...,1:-1], F.conv2d(dx, Dx, padding=1)[...,1:-1,:]
            div = (dy + dx)[0,0]
            lap = (ddx + ddy)[0,0]
            div.div_(abs(div.max()))
            lap.div_(abs(lap.max()))
            divs.append(div)
            laps.append(lap)

        laps = (torch.stack(laps, 0))
        laps = torch.stack((
            (-laps).clamp(0,1000),
            torch.zeros_like(laps),
            ( laps).clamp(0,1000)), -1) # NHW -> NHW3
        divs = (torch.stack(divs, 0))
        divs = torch.stack((
            (-divs).clamp(0,1000),
            torch.zeros_like(divs),
            ( divs).clamp(0,1000)), -1) # NHW -> NHW3

        print(laps.view(laps.size(0),-1).sum(1)) # Sum is preserved
        print('lap nonzeros per iter', (laps>0).view(laps.size(0),-1).sum(1))
        print('div nonzeros per iter', (divs>0).view(divs.size(0),-1).sum(1))

        for name, arr in zip(('(scalar) div of box', 'lap box'), (divs, laps)):
            arr = (arr.cpu().numpy() * 255).astype(np.uint8)
            arr = np.hstack(arr)
            arr = cv2.resize(arr, (0,0), fx=16, fy=16, interpolation=cv2.INTER_NEAREST)
            cv2.imshow(name, arr)

    # Show manhattan box blur. Not sure canonical name.
    if 1:
        print('\nDoing 2d manhattan blur.')
        hw = 16
        img = torch.zeros((hw,hw), dtype=torch.float32).cuda()
        img[hw//2, hw//2] = 1
        imgs = [img]
        kernel = torch.zeros((1,1,3,3)).cuda()
        for dy in range(-1,2):
            for dx in range(-1,2):
                if dy == 0 or dx == 0:
                    kernel[:,:,dy+1,dx+1] = 1
        kernel.div_(kernel.sum())
        for i in range(3):
            img = F.conv2d(img.unsqueeze(0).unsqueeze(0), kernel, stride=1, padding=1)[0,0]
            imgs.append(img)
        imgs = (torch.stack(imgs, 0))

        print(imgs.view(imgs.size(0),-1).sum(1)) # Sum is preserved
        print('nonzeros per iter', (imgs>0).view(imgs.size(0),-1).sum(1))
        # imgs.sqrt_() # Easier to see.

        imgs = (imgs.cpu().numpy() * 255).astype(np.uint8)
        imgs = np.hstack(imgs)
        imgs = cv2.resize(imgs, (0,0), fx=16, fy=16, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('manhattan box', imgs)

    # Compute regular and manhattan box blur.
    if 1:
        print('\nDoing 3d regular blur.')
        hw = 24
        img = torch.zeros((hw,hw,hw), dtype=torch.float32).cuda()
        img[hw//2, hw//2, hw//2] = 1
        imgs = [img]
        kernel = torch.zeros((1,1,3,3,3)).cuda()
        for dz in range(-1,2):
            for dy in range(-1,2):
                for dx in range(-1,2):
                    kernel[:,:,dz+1,dy+1,dx+1] = 1
        kernel.div_(kernel.sum())
        numberConvolvedNonzeros = [1]
        for i in range(3):
            img = F.conv3d(img.unsqueeze(0).unsqueeze(0), kernel, stride=1, padding=1)
            numberConvolvedNonzeros.append((F.conv3d(img, img, padding=hw//2) > 0).sum().item())
            img = img[0,0]
            imgs.append(img)
        imgs = (torch.stack(imgs, 0))

        print(imgs.view(imgs.size(0),-1).sum(1)) # Sum is preserved
        print('nonzeros per iter', (imgs>0).view(imgs.size(0),-1).sum(1))
        print('number non-zero in self-convolutions', numberConvolvedNonzeros)

        print('\nDoing 3d manhattan blur.')
        img = torch.zeros((hw,hw,hw), dtype=torch.float32).cuda()
        img[hw//2, hw//2, hw//2] = 1
        imgs = [img]
        kernel = torch.zeros((1,1,3,3,3)).cuda()
        for dz in range(-1,2):
            for dy in range(-1,2):
                for dx in range(-1,2):
                    if dy == 0 or dx == 0:
                        kernel[:,:,dz+1,dy+1,dx+1] = 1
        kernel.div_(kernel.sum())
        numberConvolvedNonzeros = [1]
        for i in range(3):
            img = F.conv3d(img.unsqueeze(0).unsqueeze(0), kernel, stride=1, padding=1)
            numberConvolvedNonzeros.append((F.conv3d(img, img, padding=hw//2) > 0).sum().item())
            img = img[0,0]
            imgs.append(img)
        imgs = (torch.stack(imgs, 0))

        print(imgs.view(imgs.size(0),-1).sum(1)) # Sum is preserved
        print('nonzeros per iter', (imgs>0).view(imgs.size(0),-1).sum(1))
        print('number non-zero in self-convolutions', numberConvolvedNonzeros)

    cv2.waitKey(0)

# def get_stencil(iteration=2, kind='manhattan', size=-1, normalize='l1'):
def get_stencil(iteration=2, kind='regular', size=-1, normalize='l1'):
    if size < 0: size = (iteration)*2 + 1

    img = torch.zeros((1,1,size,size,size), dtype=torch.float32).cuda()
    img[:, :, size//2, size//2, size//2] = 1
    kernel = torch.zeros((1,1,3,3,3)).cuda()
    for dz in range(-1,2):
        for dy in range(-1,2):
            for dx in range(-1,2):
                if kind == 'manhattan':
                    if dy == 0 or dx == 0:
                        kernel[:,:,dz+1,dy+1,dx+1] = 1
                elif kind == 'regular':
                        kernel[:,:,dz+1,dy+1,dx+1] = 1
                else:
                    assert False

    for i in range(iteration):
        img = F.conv3d(img, kernel, stride=1, padding=1)

    if normalize == 'l1': img.div_(img.sum())
    elif normalize == 'l2': img.div_(img.norm())
    elif normalize == 'none' or normalize == None: pass
    else: assert False

    return img[0,0]

__lap_kernels = {}
def get_laplacian(A):
    global __lap_kernels
    d = A.ndim
    A = A.unsqueeze(0).unsqueeze_(0)
    if d == 2:
        if 2 not in __lap_kernels:
            K = torch.zeros((3,)*d, device=A.device).unsqueeze_(0).unsqueeze_(0)
            K[...,1,1] = -4
            for dy in range(3):
                for dx in range(3):
                    if (dx == 1 and dy != 1) or (dy == 1 and dx != 1):
                        K[...,dy,dx] = 1
            __lap_kernels[2] = K
        else:
            K = __lap_kernels[2]
        return F.conv2d(A, K, padding=2)
    if d == 3:
        if 3 not in __lap_kernels:
            K = torch.zeros((3,)*d, device=A.device).unsqueeze_(0).unsqueeze_(0)
            K[...,1,1,1] = -6
            for dz in range(3):
                for dy in range(3):
                    for dx in range(3):
                        if (dz != 1 and dy == 1 and dx == 1) or (dy != 1 and dx == 1 and dz == 1) or (dx != 1 and dz == 1 and dy == 1):
                            K[...,dz,dy,dx] = 1
            __lap_kernels[3] = K
        else:
            K = __lap_kernels[3]
        return F.conv3d(A, K, padding=2)[0,0]
    assert False
def get_convolved_lap(A):
    d = A.ndim
    L = get_laplacian(A).unsqueeze_(0).unsqueeze_(0)
    A = A.unsqueeze(0).unsqueeze_(0)
    if d == 2: return F.conv2d(L,A,padding=4)[0,0]
    if d == 3: return F.conv3d(L,A,padding=4)[0,0]
    assert False
if 0:
    s = get_stencil()
    print(' - A')
    print(s,s.shape)
    l = get_laplacian(s)
    print(' - Laplacian[A]')
    print(l, l.shape)
    print(' - Laplacian[A] * A')
    la = get_convolved_lap(s)
    print(la, la.shape)
    exit()

__grad_kernels = {}
def get_grad(A):
    global __grad_kernels
    d = A.ndim
    A = A.unsqueeze(0).unsqueeze_(0)
    if d == 2:
        if 2 not in __grad_kernels:
            K = torch.zeros((3,)*d, device=A.device).unsqueeze_(0).unsqueeze_(0).repeat(3,1,1,1)
            K[...,1,1,1] = 2

            K[0,0,0,1] = -1
            K[0,0,2,1] = -1

            K[1,0,1,0] = -1
            K[1,0,1,2] = -1
            __grad_kernels[2] = K
        else:
            K = __grad_kernels[2]
        return F.conv2d(A, K, padding=2)[0]
    if d == 3:
        if 3 not in __grad_kernels:
            K = torch.zeros((3,)*d, device=A.device).unsqueeze_(0).unsqueeze_(0).repeat(3,1,1,1,1)
            K[...,1,1,1] = 2

            K[0,0,0,1,1] = -1
            K[0,0,2,1,1] = -1

            K[1,0,1,0,1] = -1
            K[1,0,1,2,1] = -1

            K[2,0,1,1,0] = -1
            K[2,0,1,1,2] = -1
            __grad_kernels[3] = K
        else:
            K = __grad_kernels[3]
        return F.conv3d(A, K, padding=2)[0]
    assert False


def forward_into_grid(x, D, pad, relativePad=.05):
    assert x.ndim == 2 and x.size(1) == 3

    size = (1<<D)

    # FIXME: Probably remove relative pad.
    extent = (x.max() - x.min())
    relativePadAdd = int((1<<D) * relativePad)
    pad0 = pad
    pad = pad0 + relativePadAdd

    scale = ((1<<D) - 2*pad) / extent
    off = -x.min(0).values + pad/scale

    print(f' - From extent {extent}, pad {pad0}, relative pad {relativePad} -> {relativePadAdd}, totalPad {pad}')
    print(f' - Have D {D} -> ncell {(1<<D)}')
    print(f' - Have offset {off}')
    print(f' - Have scale  {scale}')

    x = x + off.view(1,3)
    x.mul_(scale)

    print(f' - With transformed points spanning:')
    print(f' -        {x.min(0).values}')
    print(f' -     -> {x.max(0).values}')

    assert (x>=0).all()
    assert (x<=size).all()
    return x, off, scale, size

def backward_from_grid(x, off, scale):
    x = x / scale
    x.add_(off)
    return x

def iter_stencil(st):
    return zip(st.indices().t() - st.size()[0]//2, st.values())
def iter_stencil_nonzero(st):
    return ((D,W) for (D,W) in zip(st.indices().t() - st.size()[0]//2, st.values()) if abs(W)>1e-8)

def form_solve_system(stencil, coo, v, SO, dev):
    st = time.time()

    convLapStencil = get_convolved_lap(stencil)
    convLapStencilSt = convLapStencil.to_sparse().coalesce()
    Lc,Lv = torch.empty((2,0),dtype=torch.long,device=dev), torch.empty((0,),dtype=torch.float32,device=dev)
    print(' - convLapStencil nnz', convLapStencilSt._nnz())
    # Add one to value so that we can detect invalid matches later (as 0) and mask them out.
    # indexLookupO = torch.sparse_coo_tensor(st1.indices(), 1+torch.arange(st1._nnz(),device=dev), size=V.size()[:3]).coalesce()
    indexLookupO = torch.sparse_coo_tensor(coo, 1+torch.arange(coo.size(1),device=dev), size=(SO,)*3).coalesce()

    for D,W in iter_stencil_nonzero(convLapStencilSt):
        tmpc = coo - D.view(3,1)
        tmpv = torch.ones_like(tmpc[0])

        if 1:
            if 0:
                # Ensure we have *exact* same layout by adding zeros with original coordinates.
                # Required because without this we may have < |O| entries.
                tmpc = torch.cat((tmpc, coo), 1)
                tmpv = torch.cat((tmpv, torch.zeros_like(coo[0])), 0)
                tmp = torch.sparse_coo_tensor(tmpc,tmpv, size=indexLookupO.size()).coalesce()
                tmp_v = (tmp * indexLookupO).coalesce()

            # This appears to do it, tbh I don't understand why.
            if 1:
                tmp = torch.sparse_coo_tensor(tmpc,tmpv, size=indexLookupO.size()).coalesce()
                tmp_v = (tmp * indexLookupO).coalesce()
                tmp_v = (tmp_v + torch.sparse_coo_tensor(tmpc, torch.zeros_like(coo[0]), size=indexLookupO.size())).coalesce()

            # if tmp_v.values().sum() != 0: print(f' - tmp_v:\n {tmp_v} at D={D} with W={W}')

            assert tmp_v._nnz() == SO # See above warning.
            row_ids = torch.arange(SO, device=dev)
            col_ids = tmp_v.values()
            mask = col_ids > 0
            row_ids = row_ids[mask]
            col_ids = col_ids[mask] - 1
        else:
            tmp1 = torch.sparse_coo_tensor(tmpc,tmpv, size=indexLookupO.size()).coalesce()
            tmp2 = torch.sparse_coo_tensor(tmpc,torch.arange(len(tmpc[0])), size=indexLookupO.size()).coalesce()

        if 0 and len(row_ids) > 0 and not (D==0).all():
            print(f' - Adding off-diagonal pairs (W={W.item()}):')
            print('        ', row_ids)
            print('        ', col_ids)

        Lc = torch.cat((Lc, torch.stack((row_ids, col_ids), 0)), 1)
        Lv = torch.cat((Lv, W.repeat(len(col_ids))), 0)

    L = torch.sparse_coo_tensor(Lc, Lv, size=(SO,SO)).coalesce()
    print(' - L:\n', L)

    # -----------------------------------------------------------------------------------------------------------
    # Solve
    #

    # Note: just one level for now..
    print(' - L shape',L.shape, 'nnz', L._nnz())
    x0 = torch.zeros(SO, device=dev)

    # x1 = solve_cg(L, v, x0, T=3, preconditioned=True, debug=True)
    x1 = solve_cg(L, v, x0, T=1000, preconditioned=True, debug=True)
    # print(x1)

    print(f' - form_solve_system took {time.time()-st:.2f}')
    return x1


# Avoid forming the product J'J, instead just keep J and apply J'(Jx) in the conjugate gradient steps.
# In case of the 3d 5x5 stencil, this is a memory savings of about 6x.
# The laplacian stencil itself has a support of 125 cells.
# The laplacian stencil convolved with itself has a support of 729 cells.
#
# The body of this function is much like form_solve_system(), only
# it DOES NOT use convLapStencil.
#
# WARNING: I think this is incorrect because it's possible that the gradient in both sides of the outer inner product
# expand the support of the arguments. This function will miss those created non-zeros?
def form_solve_system_AtA(stencil, coo, v, SO, dev):
    st = time.time()

    lapStencil = get_laplacian(stencil)
    lapStencilSt = lapStencil.to_sparse().coalesce()

    Lc,Lv = torch.empty((2,0),dtype=torch.long,device=dev), torch.empty((0,),dtype=torch.float32,device=dev)
    print(' - lapStencil nnz', lapStencilSt._nnz())
    # Add one to value so that we can detect invalid matches later (as 0) and mask them out.
    # indexLookupO = torch.sparse_coo_tensor(st1.indices(), 1+torch.arange(st1._nnz(),device=dev), size=V.size()[:3]).coalesce()
    indexLookupO = torch.sparse_coo_tensor(coo, 1+torch.arange(coo.size(1),device=dev), size=(SO,)*3).coalesce()

    for D,W in iter_stencil_nonzero(lapStencilSt):
        tmpc = coo - D.view(3,1)
        tmpv = torch.ones_like(tmpc[0])

        if 1:
            if 0:
                # Ensure we have *exact* same layout by adding zeros with original coordinates.
                # Required because without this we may have < |O| entries.
                tmpc = torch.cat((tmpc, coo), 1)
                tmpv = torch.cat((tmpv, torch.zeros_like(coo[0])), 0)
                tmp = torch.sparse_coo_tensor(tmpc,tmpv, size=indexLookupO.size()).coalesce()
                tmp_v = (tmp * indexLookupO).coalesce()

            # This appears to do it, tbh I don't understand why.
            if 1:
                tmp = torch.sparse_coo_tensor(tmpc,tmpv, size=indexLookupO.size()).coalesce()
                tmp_v = (tmp * indexLookupO).coalesce()
                tmp_v = (tmp_v + torch.sparse_coo_tensor(tmpc, torch.zeros_like(coo[0]), size=indexLookupO.size())).coalesce()

            # if tmp_v.values().sum() != 0: print(f' - tmp_v:\n {tmp_v} at D={D} with W={W}')

            assert tmp_v._nnz() == SO # See above warning.
            row_ids = torch.arange(SO, device=dev)
            col_ids = tmp_v.values()
            mask = col_ids > 0
            row_ids = row_ids[mask]
            col_ids = col_ids[mask] - 1
        else:
            tmp1 = torch.sparse_coo_tensor(tmpc,tmpv, size=indexLookupO.size()).coalesce()
            tmp2 = torch.sparse_coo_tensor(tmpc,torch.arange(len(tmpc[0])), size=indexLookupO.size()).coalesce()

        if 0 and len(row_ids) > 0 and not (D==0).all():
            print(f' - Adding off-diagonal pairs (W={W.item()}):')
            print('        ', row_ids)
            print('        ', col_ids)

        Lc = torch.cat((Lc, torch.stack((row_ids, col_ids), 0)), 1)
        Lv = torch.cat((Lv, W.repeat(len(col_ids))), 0)

    L = torch.sparse_coo_tensor(Lc, Lv, size=(SO,SO)).coalesce()
    print(' - L:\n', L)

    # -----------------------------------------------------------------------------------------------------------
    # Solve
    #

    # Note: just one level for now..

    print(' - L shape',L.shape, 'nnz', L._nnz())
    x0 = torch.zeros(SO, device=dev)

    # x1 = solve_cg(L, v, x0, T=3, preconditioned=True, debug=True)
    x1 = solve_cg_AtA(L, v, x0, T=1000, preconditioned=True, debug=True)

    print(f' - form_solve_system_AtA took {time.time()-st:.2f}')
    return x1

def compute_divV(st0, V, stencil_st):
    dev = V.device
    size = V.size(0)
    coo = st0.indices()

    # Compute div[V]
    # NOTE: This relies on linearity. It computes the divergnce by a sparse 3d conv (not memory efficient)
    # NOTE: There are some tricks I see to improve efficiency here (using sparse tensor shifts by multiplying
    #                                                               that could reduce mem consumption)
    # WARNING: This is a convolution of a convolution. It requires even more padding of the input grid to
    #          prevent invalid indices. Instead of that, I'll mask out invalid entries.
    #          This also increases memory usage upto 6x!
    #          But we only need these data at certain points (those indices occupied by V)
    #          You could form a sparse tensor at each iter and multiply to keep only needed entries OR
    #          use the other method of computing that does not have this issue...
    # FIXME: going to need the shift/multiply based approach, this hits the worse case 6x (of the already 125x V).
    #        that is RAM usage upto 750x initial point set...
    # NOTE: There may even be a better way that what I had in mind by casting as sparse matrix multiply...
    Vc2 = V.indices()
    Vv2 = V.values()
    # The divergence of a point sample is zero at the point, but non-zero just around it.
    if 0:
        # Not sure if works, but next branch does
        divVc,divVv = torch.empty((3,0),dtype=torch.long,device=dev), torch.empty((0),dtype=torch.float32,device=dev)
        def collapse_divV():
            nonlocal divVc, divVv
            mask  = (divVc>=0).all(0) & (divVc<size).all(0)
            divVc = divVc[:,mask]
            divVv = divVv[mask]
            divV  = torch.sparse_coo_tensor(divVc,divVv, size=(size,size,size,1)).coalesce()
            divVc,divVv = divV.indices(), divV.values()
            return divV
        for d in range(3):
            d = torch.LongTensor((d==0,d==1,d==2)).view(3,1)
            divVc = torch.cat((divVc, Vc2 - D.view(3,1)), 1)
            divVv = torch.cat((divVv, -Vv2.sum(1,keepdim=True)), 0)
            divVc = torch.cat((divVc, Vc2 + D.view(3,1)), 1)
            divVv = torch.cat((divVv, Vv2.sum(1,keepdim=True)), 0)
            collapse_divV() # This is optional. Trades less speed for less RAM usage
        divV = collapse_divV()
        print(f' - divV sizes (uncoalesced {divVc.size(1)}) (final nnz {divV._nnz()})')
        # Sub-select divV to have *EXACT* same indices as V
        divV = divV * torch.sparse_coo_tensor(V.indices(), torch.ones_like(V.values()[:,0:1]), size=(*V.size()[:3],1))
        print(f' - divV subselected nnz {divV._nnz()}')
    else:
        accessor = torch.sparse_coo_tensor(Vc2, torch.ones (Vc2.size(1),device=dev), size=V.size()[:3]).coalesce()
        divV     = torch.sparse_coo_tensor(Vc2, torch.zeros(Vc2.size(1),device=dev), size=V.size()[:3]).coalesce()
        for d in range(3):
            dd = torch.LongTensor((d==0,d==1,d==2)).view(3,1).to(dev)
            if 0:
                left  = (torch.sparse_coo_tensor(Vc2 + dd.view(3,1), Vv2[:,d], size=V.size()[:3]).coalesce() * accessor).coalesce()
                right = (torch.sparse_coo_tensor(Vc2 - dd.view(3,1), Vv2[:,d], size=V.size()[:3]).coalesce() * accessor).coalesce()
            else:
                l = Vc2 + dd.view(3,1)
                r = Vc2 - dd.view(3,1)
                lv = Vv2[(l>=0).all(0) & (l<size).all(0)][:,d]
                rv = Vv2[(r>=0).all(0) & (r<size).all(0)][:,d]
                # lv = Vv2[:,d]
                # rv = Vv2[:,d]
                left  = (torch.sparse_coo_tensor(l, lv, size=V.size()[:3]).coalesce() * accessor).coalesce()
                right = (torch.sparse_coo_tensor(r, rv, size=V.size()[:3]).coalesce() * accessor).coalesce()
                divV += (right - left).coalesce()
        divV = divV.coalesce()
        # Leave the zeros.
        # mask = divV.values() != 0
        # divV = torch.sparse_coo_tensor(divV.indices()[:,mask],divV.values()[mask])
        print('V:\n',V)
        print('divV:\n',divV)
        print(' -     V  shape', V._nnz())
        print(' - div[V] shape', divV._nnz())

    return divV


def compute_v(st0,V,stencil_st):
    # Compute vector v, with each coordinate v_o = <div[V] , Fo>

    divV = compute_divV(st0, V, stencil_st)

    dev = V.device
    coo = st0.indices()



    # In this section F will need to be a matrix that maps:
    #      from the size of div[V] (same as size of V, upto 125x the size of octree)
    #      to the size of O (the octree)

    # NOTE: to compute the v vector:
    #     1) Form sparse matrix Fv which maps R^|V| -> R^|O|
    #          a) form Vind. Has same indices as V, but values as arange(len(V))
    #          b) form Oones, indices same as O, but values as ones.
    #          c) do 3d sparse conv, shifting with Vind, weighing with W, and masking with Oones
    #              i) Add columns/rows of final matrix each iter.
    #             ii) The matrix is exactly |O| rows and has |V| columns, but only upto 125 entries per row.

    # NOTE: There may be a better way to do this using the sparse matrix approach used below (rather than a sparse tensor used here).
    # TODO: Implement above using sparse matrix technique

    SV = V._nnz()
    SO = st0._nnz()

    Fv_indices,Fv_values = torch.empty((2,0),dtype=torch.long,device=dev), torch.empty((0,),dtype=torch.float32,device=dev)
    indexLookupDV = torch.sparse_coo_tensor(divV.indices(), torch.arange(divV._nnz(),device=dev), size=divV.size()[:3]).coalesce()
    indexLookupO = torch.sparse_coo_tensor(st0.indices(), torch.arange(st0._nnz(),device=dev), size=V.size()[:3]).coalesce()
    # WARNING: This requires tmp_v to have the same indices layout after sorting as before. So all indices must be resident, and have there orders unchanged.
    #          This *should* work as long as stencil shift D is the same for every element (it is) and it causes no wrap around nor dropping (we have to pad enough)
    for D,W in iter_stencil(stencil_st):
        # tmpc = torch.cat((tmpc, coo2 - D.view(3,1)), 1)
        # tmpv = torch.cat((tmpv, nrmls2), 0)
        tmpc = coo - D.view(3,1)
        tmpv = torch.ones_like(tmpc[0])
        tmp = torch.sparse_coo_tensor(tmpc,tmpv, size=indexLookupDV.size()).coalesce()
        tmp_v = (tmp * indexLookupDV).coalesce()
        # tmp_c = (tmp * indexLookupO).coalesce()
        # tmp_c = (torch.sparse_coo_tensor(tmp_v.indices(),torch.ones_like(tmp_v.indices()[0]), size=indexLookupDV.size()).coalesce() * indexLookupO).coalesce()
        # row_ids = tmp_c.values()
        assert tmp_v._nnz() == SO # See above warning.
        row_ids = torch.arange(SO, device=dev)
        col_ids = tmp_v.values()
        '''
        if row_ids.numel() > 0 and col_ids.numel() > 0 :
            print(f' - D={D} got min/max ids (row = {row_ids.min()}/{row_ids.max()} sz {row_ids.size()}) (col = {col_ids.min()}/{col_ids.max()} sz {col_ids.size()})')
        else:
            print(f' - D={D} got min/max ids (row sz {row_ids.size()}) (col = sz {col_ids.size()})')
        '''
        # print(f' - Sizes (col = {col_ids.size()}) (row = {row_ids.size()}) (tmpc = {tmpc.size()})')
        Fv_indices = torch.cat((Fv_indices, torch.stack((row_ids, col_ids), 0)), 1)
        Fv_values = torch.cat((Fv_values, W.repeat(SO)), 0)

        if 0:
            # This shows that under if the two conditions are met, the index order should not change
            tmpc = coo - D.view(3,1)
            tmpv = torch.arange(len(tmpc[0]), device=dev)
            tmp = torch.sparse_coo_tensor(tmpc,tmpv, size=indexLookupDV.size()).coalesce()

    Fv = torch.sparse_coo_tensor(Fv_indices, Fv_values, size=(SO,SV)).coalesce()
    print(' - Fv:\n', Fv)
    v = Fv @ divV.values()
    return v


# Sparse 3d convolution.
def run_psr(pts0, nrmls0, D=6):
    dev = pts0.device

    # print(f' - stencil {stencil.shape}')
    print(f' - pts {pts.shape}')
    print(f' - nrmls {nrmls.shape}')

    # Map to fixed resolution grid, then de-duplicate entries.
    coo0, off, scale, size = forward_into_grid(pts0, D=D, pad=9)
    st0 = torch.sparse_coo_tensor(coo0.long().t(), nrmls0, size=(size,size,size,3)).coalesce()
    # st0._values(0.div_(st0._values().norm(dim=1,keepdim=True))
    # NOTE: Consider not normalizing, that we we aren't merging bins and also losing samples.
    coo, nrmls2 = st0.indices(), F.normalize(st0.values(), dim=1)

    stencil = get_stencil()
    stencil_st = stencil.to_sparse().coalesce() # A convenient way to iterate over indices/values

    # -----------------------------------------------------------------------------------------------------------
    # Compute vector field V
    #
    Vc,Vv = torch.empty((3,0),dtype=torch.long,device=dev), torch.empty((0,3),dtype=torch.float32,device=dev)

    for D,W in iter_stencil(stencil_st):
        Vc = torch.cat((Vc, coo - D.view(3,1)), 1)
        Vv = torch.cat((Vv, W*nrmls2), 0)

    V = torch.sparse_coo_tensor(Vc,Vv, size=(size,size,size,3)).coalesce()
    print(f' - V sizes (uncoalesced {Vc.size(1)}) (final nnz {V._nnz()}={V.indices().size(1)})')

    # -----------------------------------------------------------------------------------------------------------
    # Compute target `b` vector, whose components are the inner product of the basis functions Fo
    # with the divergence of V
    #
    # The simplest way to do this would be to compute `div[V]` then convolve with Fo.
    # That requires as much ram as V. However V is much larger than {Fo}.
    # So (perhaps?) another way to compute it is by considering Fo as the main thing.
    #
    # Here is the first one.

    SV = V._nnz() # basis support
    SO = st0._nnz() # Octree size

    v = compute_v(st0, V, stencil_st)
    print(' - v:', v.shape, '\n', v)

    # -----------------------------------------------------------------------------------------------------------
    # Compute modified Laplacian matrix L
    #


    # It is now necessary to use actual sparse *matrices* instead of using st's as arrays.
    # div[V] will be a matrix. We'll select only the entries from the tensor that correspond to nodes {o}.
    # Fo will be a sparse matrix.

    # Khazdan writes the matrix has upto 125 nonzero columns, but shouldn't it really be <125^2 because
    # a convolution of the two node functions results in a much bigger support region than a convolution of a node fn and a point?

    # Don't I need to convolve the stencil with *itself*, not with the base function?
    # Like if it is convolved thrice (*3) convolving with itself is like squaring it to get (*6).
    # This is also how you could compute the L matrix, use the self-convolved stencil just like the normal one.
    #
    # my 'convLapStencil' has 1215 nonzeros, which determines the max number of nnz columns/rows in L.
    #

    # FIXME: Actually you can use the eqn from the second paper to see how A_ij is like J'J, and using conjugate gradient algo,
    #        you needn't explicilty form it.
    #        The J matrix is the grad_x+grad_y+grad_z.
    #        The sizes of J and J'J are the same, but J has much less non-zeros!

    # The second call is more sparse, but slower on small inputs.
    # It requires 2x sparse matrix multiplies, but about 10x less memory.
    # x1 = form_solve_system(stencil, coo, v, SO, dev)
    x1 = form_solve_system_AtA(stencil, coo, v, SO, dev)


    # -----------------------------------------------------------------------------------------------------------
    # Polygonalization
    #

    Ic,Iv = torch.empty((3,0),dtype=torch.long,device=dev), torch.empty((0),dtype=torch.float32,device=dev)
    for D,W in iter_stencil(stencil_st):
        Ic = torch.cat((Ic, coo - D.view(3,1)), 1)
        Iv = torch.cat((Iv, torch.full((SO,),W.item(),dtype=torch.float32,device=dev)), 0)

    I = torch.sparse_coo_tensor(Ic,Iv, size=(size,size,size)).coalesce()
    print(f' - I sizes (uncoalesced {Ic.size(1)}) (final nnz {I._nnz()}={I.indices().size(1)})')

    positions, inds, marchedOffScale = run_marching(I, isolevel=.05)
    pts1 = (pts0 + off) * scale
    pts1 = (pts1 + marchedOffScale[0]) * marchedOffScale[1]
    show_marching_viz(positions, inds, pts=pts1,nrls=nrmls0)




if __name__ == '__main__':
    torch.manual_seed(0)
    if 0:
        pts   = torch.randn(512*16, 3).cuda()
        # pts   = pts / pts.norm(dim=1,keepdim=True)
        pts   = pts / pts.abs().sum(dim=1,keepdim=True)
        pts   = pts * torch.rand(pts.size(0),1,device=pts.device).sqrt() # Random inside sphere
        pts = pts * .5
        nrmls   = pts / pts.norm(dim=1,keepdim=True)
    else:
        ps     = torch.randn(512*2, 3).cuda()
        ps     = ps / ps.abs().sum(dim=1,keepdim=True)
        ps     = ps * torch.rand(ps.size(0),1,device=ps.device).sqrt() # Random inside sphere
        ps     = ps * .25
        nrmls0 = ps / ps.norm(dim=1,keepdim=True)
        ps0    = ps - .2

        ps1    = ps + .2
        nrmls1 = ps / ps.norm(dim=1,keepdim=True)
        pts    = torch.cat((ps0, ps1), 0)
        nrmls  = torch.cat((nrmls0, nrmls1), 0)
    print(' - pts  :\n', pts)
    print(' - nrmls:\n', nrmls)
    run_psr(pts, nrmls)
    # viz_conv_of_lap()
    viz_box()
