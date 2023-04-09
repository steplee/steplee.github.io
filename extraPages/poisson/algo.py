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


# Sparse 3d convolutions.
def form_system(pts, nrmls):
    stencil = get_stencil()
    S = stencil.size(0)

    # Iterate over the stencil, replicate
    stencil = stencil.to_sparse().coalesce()
    for D,W in zip(stencil.indices().t(), stencil.values()):
        print(D,W)




# Borke marching cubes/tetra needs array of gridcells.
# Each gridcell contains data about 8 corners
# This is just a sparse conv
def replicate_to_gridcells(sparseTensor):
    c = sparseTensor.indices()
    v = sparseTensor.values()
    D = c.device

    N = c.size(1)
    # Dense-dim + sparse-dim
    size = (*sparseTensor.size(), 8)
    ov = torch.zeros((N,8), device=D)
    out = torch.sparse_coo_tensor(c,ov,size=size).coalesce()

    stencil = torch.sparse_coo_tensor(c, torch.ones_like(ov), size=size).coalesce()

    grid = [
        (0,0,0), (1,0,0), (1,1,0), (0,1,0),
        (0,0,1), (1,0,1), (1,1,1), (0,1,1)]
    # grid = [
        # (0,0,0), (1,0,0), (1,0,1), (0,0,1),
        # (0,1,0), (1,1,0), (1,1,1), (0,1,1)]
    # grid = [
        # (0,0,1), (1,0,1), (1,0,0), (0,0,0),
        # (0,1,1), (1,1,1), (1,1,0), (0,1,0)]

    # for dz in range(2):
        # for dy in range(2):
            # for dx in range(2):
                # idx = dz*4+dy*2+dx
    for idx, (dx,dy,dz) in enumerate(grid):
                # off = torch.LongTensor([dz,dy,dx]).to(c.device).view(3,1)
                off = torch.LongTensor([dx,dy,dz]).to(c.device).view(3,1) # FIXME: Is this right?

                coff = c.clone() - off
                # coff = c.clone() + off
                voff = ov.clone()
                voff[...,idx] = v
                mask  = (coff>=0).all(0)
                mask &= (coff<torch.LongTensor(list(sparseTensor.size())).to(D)[:3].view(3,1)).all(0)
                voff = voff[mask]
                coff = coff[:,mask]
                toff = stencil * torch.sparse_coo_tensor(coff, voff, size=size).coalesce()

                out += toff

                # Use this if running out of RAM
                # out = out.coalesce()

    out = out.coalesce()
    return out
def test_replicate():
    coo = torch.cuda.FloatTensor([
        0,0,0,
        0,1,0,
        0,1,1,
        0,0,1,

        1,5,1,
        2,5,1,
        1,6,1,
        ]).view(-1,3).t().contiguous()
    val = torch.cuda.FloatTensor([1,2,3,4, 1,1,1])

    coo = torch.cuda.FloatTensor([
        4,4,4,

        4,4,4+1,
        4+1,4,4+1,
        4+1,4,4,
        4,4,4,
        4,4+1,4+1,
        4+1,4+1,4+1,
        4+1,4+1,4,
        4,4+1,4,
        ]).view(-1,3).t().contiguous()
    val = torch.cuda.FloatTensor([0, 1,2,3,4,5,6,7,8])

    st0 = torch.sparse_coo_tensor(coo,val, size=(10,10,10)).coalesce()

    st1 = replicate_to_gridcells(st0)
    print('Initial:\n', st0)
    print('Final:\n', st1)
    sys.exit(0)
# test_replicate()


# Returns up to two tris per input.
# Should be called 6 times per grid cell
def marching_tetra_tri(positions, vals, scale,  iso, v0,v1,v2,v3):
    N = positions.size(0)
    assert positions.size(1) == 3
    assert vals.size() == (N,8)
    D = positions.device

    # Must be long -- cannot be byte. That gets interpreted as mask.
    triIndex  = torch.zeros((N), dtype=torch.int64, device=D)
    triIndex |= (1) * (vals[:,v0] < iso).long()
    triIndex |= (2) * (vals[:,v1] < iso).long()
    triIndex |= (4) * (vals[:,v2] < iso).long()
    triIndex |= (8) * (vals[:,v3] < iso).long()
    print(' - triIndex shape',triIndex.shape)
    # print(triIndex)

    # This replaces the switch statement in `PolygoniseTri`
    table1 = {}
    # table1[0x0] = table1[0xF] = None
    table1[0x1] = table1[0xE] = (v0,v1, v0,v2, v0,v3)
    table1[0x2] = table1[0xD] = (v1,v0, v1,v3, v1,v2)
    table1[0x3] = table1[0xC] = (v0,v3, v0,v2, v1,v3)
    table1[0x4] = table1[0xB] = (v2,v0, v2,v1, v2,v3)
    table1[0x5] = table1[0xA] = (v0,v1, v2,v3, v0,v3)
    table1[0x6] = table1[0x9] = (v0,v1, v1,v3, v2,v3)
    table1[0x7] = table1[0x8] = (v3,v0, v3,v2, v3,v1)

    table2 = {}
    table2[0x3] = table2[0xC] = (v1,v3, v1,v2, v0,v2)
    table2[0x5] = table2[0xA] = (v0,v1, v1,v2, v2,v3)
    table2[0x6] = table2[0x9] = (v0,v1, v0,v2, v2,v3)


    p = torch.FloatTensor([
        0,0,0, 1,0,0, 1,1,0, 0,1,0,
        0,0,1, 1,0,1, 1,1,1, 0,1,1]).to(D).reshape(-1,3) * scale
        # 0,0,0, 1,0,0, 1,0,1, 0,0,1,
        # 0,1,0, 1,1,0, 1,1,1, 0,1,1]).to(D).reshape(-1,3) * scale
    # grid_ = [ (0,0,1), (1,0,1), (1,0,0), (0,0,0), (0,1,1), (1,1,1), (1,1,0), (0,1,0)]
    # p = torch.FloatTensor(grid_).to(D).reshape(-1,3) * scale
    # p[...,2] *= -1

    # p[...,1] = scale - p[...,1]
    # p = p[:, [0,2,1]]

    # Shape is because:
    #      There are 16 named triangle per tetrahedron
    #      There are up to 2 triangles
    #      A triangle has 3 vertices
    #      A vertex is blended from two grid vertices
    table = torch.zeros((16,2,3,2), device=D, dtype=torch.int64) - 1
    for k,v in table1.items(): table[k, 0, :, :] = torch.LongTensor(v).to(D).view(3, 2)
    for k,v in table2.items(): table[k, 1, :, :] = torch.LongTensor(v).to(D).view(3, 2)
    # print('table\n',table)


    # Up to 2 tris each input. Each tri has 3 points. Each point has 3 coords.
    out = torch.zeros((N,2,3,3), device=D)

    def blend(ai,bi):
        valid_mask = (ai >= 0) & (bi >= 0)
        ai = ai.clamp(0,8)
        bi = bi.clamp(0,8)

        av = vals.gather(1,ai.view(-1,1))[:,0]
        bv = vals.gather(1,bi.view(-1,1))[:,0]
        ap = p[ai]
        bp = p[bi]

        mu = (iso - av) / (bv-av)
        mu.masked_fill_((iso-av).abs() < 1e-5, 0)
        mu.masked_fill_((iso-bv).abs() < 1e-5, 1)
        mu.masked_fill_((av-bv).abs() < 1e-5, 0)

        op = positions + ap + mu.view(-1,1)*(bp-ap)
        op[~valid_mask] = -1
        return op

    for triId in range(2):
        for vertexId in range(3):
            out[:,triId,vertexId,:] = blend(table[triIndex, triId, vertexId, 0], table[triIndex, triId, vertexId, 1])

    return out

# http://paulbourke.net/geometry/polygonise/source1.c
def marching_tetra(positions, vals, iso=0, gridScale=1):
    N,D = positions.size(0), positions.device


    offset = positions.min(0).values
    positions = positions.float()
    # positions.sub_(offset)
    # scale = (positions.max(0).values-positions.min(0).values) # uniform size
    # if scale == 0: scale = 1
    # positions.div_(scale)

    # vals.div_(scale)

    groups = (
        (0,2,3,7),
        (0,2,6,7),
        (0,4,6,7),
        (0,6,1,2),
        (0,6,1,4),
        (5,6,1,4),
    )

    # o = marching_tetra_tri(positions, vals, iso, 0,2,3,7)
    print(f' - positions {positions.min(0).values} -> {positions.max(0).values}')
    print(' - using grid scale', 1/gridScale)
    out = torch.zeros((N,len(groups),2,3,3), device=D)
    for i,grp in enumerate(groups):
        out[:, i] = marching_tetra_tri(positions, vals, 1/gridScale, iso, *grp)

    # Array of triangle vertex components.
    out1 = out.view(-1,9)
    N0 = out1.size(0)
    out1 = out1[(out1!=-1).all(1)]
    N1 = out1.size(0)
    out1 = out1.reshape(-1,3,3)
    print(out1)
    print(f' - removed invalid tris: {N0} -> {N1}')

    # TODO: Use sparse tensor again to reduce #verts (a lot are duplicates) and keep track of tri indices.

    out2 = out
    # out2 = out2.mul_(scale)
    # out2 = out2.add_(offset.view(1,1,3))

    from render import GridRenderer, glDisable, GL_CULL_FACE
    r = GridRenderer((1024,)*2)
    r.init(True)
    r.set_mesh(np.copy(out2.cpu().numpy(),'C'))
    glDisable(GL_CULL_FACE)

    while True:
        r.startFrame()
        r.render()
        r.endFrame()

    return out2

def test_marching_tetra():
    SZ = 8
    coords = torch.cartesian_prod(*(torch.arange(SZ),)*3)
    # vals = coords[:, 0] / coords[:,0].max()
    # vals = coords[:, 0].float() / coords[:,0].max() #- .5 # Wall
    vals = -((coords - SZ/2).float().norm(dim=1) - 3) # Sphere
    # vals = vals * .1
    # vals = (coords - (SZ+.5)/2).float().norm(dim=1) - .5

    # print('- original vals\n', vals)
    st0 = torch.sparse_coo_tensor(coords.t(),vals, size=(SZ,)*3).coalesce()
    st = replicate_to_gridcells(st0)
    coords,vals = st.indices().t(), st.values()

    print('input coords',coords.shape)
    print('input vals',vals.shape)
    print(coords[150])
    print(vals[150])
    print(coords[-230])
    print(vals[-230])
    # exit()

    # coords = coords.float() - SZ / 2


    if 0:
        grid = [
            (0,0,0), (1,0,0), (1,1,0), (0,1,0),
            (0,0,1), (1,0,1), (1,1,1), (0,1,1)]
        coords = torch.FloatTensor(grid).cuda().reshape(-1,3)
        vals = coords[:,2].reshape(1,8) + coords[:,1].reshape(1,8)
        coords = coords[0:]
        vals = vals.repeat(coords.size(0),1)

        coords = torch.cartesian_prod(*(torch.arange(SZ),)*3).reshape(-1,3).cuda()
        vals = torch.FloatTensor(grid).cuda().reshape(-1,3)
        vals = vals[:,0].unsqueeze(0).repeat(coords.size(0), 1).cuda()


    print(f' - input sizes: {coords.shape}, {vals.shape}')
    # marching_tetra(coords,vals, iso=.5)
    print('vals max',vals.max())
    print('vals min',vals.min())
    # print('- replicated coords\n', coords)
    # print('- replicated vals\n', vals)
    # exit()
    # print(vals)
    # marching_tetra(coords,vals, iso=.0)
    # vals = vals - vals.min()
    # vals = vals / vals.max()
    marching_tetra(coords,vals, iso=.4)
test_marching_tetra()
sys.exit(0)

pts   = torch.randn(512, 3).cuda()
nrmls = torch.randn(512, 3).cuda()
form_system(pts, nrmls)
viz_box()
