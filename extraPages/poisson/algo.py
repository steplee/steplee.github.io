import torch, torch.nn.functional as F
import numpy as np, sys
torch.set_printoptions(linewidth=150, edgeitems=10)

# Vizualize the box filter (iterated three times) in 2D.
# The 3d section would suggest that Khazdan 2006 is talking about the regular (non-manhattan) box blur.
# It has roughly twice the number of non-zeros.
def viz_box():
    import cv2

    # Show box blur
    if 1:
        print('\nDoing 2d regular blur.')
        hw = 16
        img = torch.zeros((hw,hw), dtype=torch.float32).cuda()
        img[hw//2, hw//2] = 1
        imgs = [img]
        for i in range(3):
            img = F.avg_pool2d(img.unsqueeze(0).unsqueeze(0), 3, 1, padding=1)[0,0]
            imgs.append(img)
        imgs = (torch.stack(imgs, 0))

        print(imgs.view(imgs.size(0),-1).sum(1)) # Sum is preserved
        print('nonzeros per iter', (imgs>0).view(imgs.size(0),-1).sum(1))
        # imgs.sqrt_() # Easier to see.

        imgs = (imgs.cpu().numpy() * 255).astype(np.uint8)
        imgs = np.hstack(imgs)
        imgs = cv2.resize(imgs, (0,0), fx=16, fy=16, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('box', imgs)

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
        hw = 16
        img = torch.zeros((hw,hw,hw), dtype=torch.float32).cuda()
        img[hw//2, hw//2, hw//2] = 1
        imgs = [img]
        kernel = torch.zeros((1,1,3,3,3)).cuda()
        for dz in range(-1,2):
            for dy in range(-1,2):
                for dx in range(-1,2):
                    kernel[:,:,dz+1,dy+1,dx+1] = 1
        kernel.div_(kernel.sum())
        for i in range(3):
            img = F.conv3d(img.unsqueeze(0).unsqueeze(0), kernel, stride=1, padding=1)[0,0]
            imgs.append(img)
        imgs = (torch.stack(imgs, 0))

        print(imgs.view(imgs.size(0),-1).sum(1)) # Sum is preserved
        print('nonzeros per iter', (imgs>0).view(imgs.size(0),-1).sum(1))

        print('\nDoing 3d manhattan blur.')
        hw = 16
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
        for i in range(3):
            img = F.conv3d(img.unsqueeze(0).unsqueeze(0), kernel, stride=1, padding=1)[0,0]
            imgs.append(img)
        imgs = (torch.stack(imgs, 0))

        print(imgs.view(imgs.size(0),-1).sum(1)) # Sum is preserved
        print('nonzeros per iter', (imgs>0).view(imgs.size(0),-1).sum(1))

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

def forward_into_grid(x, D, pad):
    assert x.ndim == 2 and x.size(1) == 3

    size = (1<<D)
    scale = ((1<<D) - 2*pad) / (x.max() - x.min())
    off = -x.min(0).values + pad/scale

    x = x + off.view(1,3)
    x.mul_(scale)

    print(x.min())
    print(x.max(), 1<<D)

    assert (x>=0).all()
    assert (x<=size).all()
    return x, off, scale, size

def backward_from_grid(x, off, scale):
    x = x / scale
    x.add_(off)
    return x


# Sparse 3d convolution.
def form_system(pts0, nrmls0, D=8):
    dev = pts0.device

    # print(f' - stencil {stencil.shape}')
    print(f' - pts {pts.shape}')
    print(f' - nrmls {nrmls.shape}')

    # Map to fixed resolution grid, then de-duplicate entries.
    coo1, off, scale, size = forward_into_grid(pts0, D=D, pad=9)
    st1 = torch.sparse_coo_tensor(coo1.long().t(), nrmls0, size=(size,size,size,3)).coalesce()
    # st1._values().div_(st1._values().norm(dim=1,keepdim=True))
    coo2, nrmls2 = st1.indices(), F.normalize(st1.values(), dim=1)

    stencil = get_stencil()
    stencil_st = stencil.to_sparse().coalesce() # A convenient way to iterate over indices/values
    iter_stencil = lambda: zip(stencil_st.indices().t(), stencil_st.values())

    # -----------------------------------------------------------------------------------------------------------
    # Compute vector field V
    #
    Vc,Vv = torch.empty((3,0),dtype=torch.long,device=dev), torch.empty((0,3),dtype=torch.float32,device=dev)

    for D,W in iter_stencil():
        Vc = torch.cat((Vc, coo2 - D.view(3,1)), 1)
        Vv = torch.cat((Vv, nrmls2), 0)

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


    # Compute vector v, with each coordinate v_o = <div[V] , Fo>

    # It is now necessary to use actual sparse *matrices* instead of using st's as arrays.
    # div[V] will be a matrix. We'll select only the entries from the tensor that correspond to nodes {o}.
    # Fo will be a sparse matrix.

    # In this section F will need to be a matrix that maps:
    #      from the size of div[V] (same as size of V, upto 125x the size of octree)
    #      to the size of O (the octree)

    # vc,vv = torch.empty((3,0),dtype=torch.long,device=dev), torch.empty((0),dtype=torch.float32,device=dev)
    # for D,W in iter_stencil():
        # Vc = torch.cat((Vc, coo2 - D.view(3,1)), 1)
        # Vv = torch.cat((Vv, nrmls2), 0)

    # NOTE: to compute the v vector:
    #     1) Form sparse matrix Fv which maps R^|V| -> R^|O|
    #          a) form Vind. Has same indices as V, but values as arange(len(V))
    #          b) form Oones, indices same as O, but values as ones.
    #          c) do 3d sparse conv, shifting with Vind, weighing with W, and masking with Oones
    #              i) Add columns/rows of final matrix each iter.
    #             ii) The matrix is exactly |O| rows and has |V| columns, but only upto 125 entries per row.






torch.manual_seed(0)
pts   = torch.randn(512, 3).cuda()
pts   = pts / pts.norm(dim=1,keepdim=True)
nrmls = pts
form_system(pts, nrmls)
viz_box()
