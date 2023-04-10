import torch, torch.nn.functional as F
import numpy as np, sys
torch.set_printoptions(linewidth=150, edgeitems=10)

# Borke marching cubes/tetra needs array of gridcells.
# Each gridcell contains data about 8 corners
# This is like a sparse conv that replicates information about its neighbors to each point.
# NOTE: This 'dualizes' the input: voxels are turned into corners of cubes.
# FIXME: Properly handle boundaries. Right now, they boundary points are implicitly given value zero.
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
    # This is more like Borke's marching cubes impl.
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

    # FIXME: Fix winding orders. I guess you have to look at isovalue and flip accordingly?

    return out

# http://paulbourke.net/geometry/polygonise/source1.c
#
# This is a batched version of the above marching tetrahedra algorithm.
#
# `positions` should be a floating point tensor [N,3] of 3d left-back-top points.
# `vals` should be a floating point tensor [N,8] of isovalues of each of the points and it's seven neighbours.
# The order should match that of `replicate_to_gridcells()`, a truth table in ZYX order.
#
def marching_tetra(positions, vals, iso=0, gridScale=1):
    N,D = positions.size(0), positions.device


    offset = positions.min(0).values
    positions = positions.float()
    # positions.sub_(offset)
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
    # print(f' - positions {positions.min(0).values} -> {positions.max(0).values}')
    # print(' - using grid scale', 1/gridScale)
    out = torch.zeros((N,len(groups),2,3,3), device=D)
    for i,grp in enumerate(groups):
        out[:, i] = marching_tetra_tri(positions, vals, gridScale, iso, *grp)

    # Array of triangle vertex components.
    out1 = out.view(-1,9)
    N0 = out1.size(0)
    out1 = out1[(out1!=-1).all(1)]
    N1 = out1.size(0)
    out1 = out1.reshape(-1,3,3)
    # print(out1)
    print(f' - removed invalid tris: {N0} -> {N1}')

    # TODO: Use sparse tensor again to reduce #verts (a lot are duplicates) and keep track of tri indices.

    out2 = out
    # out2 = out2.mul_(scale)
    # out2 = out2.add_(offset.view(1,1,3))

    # positions = np.copy(out2.cpu().numpy(),'C').astype(np.float32)
    # triPositions = np.copy(out2.cpu().numpy(),'C').astype(np.float32).reshape(-1,3,3)
    if 1:
        triPositions = out2.to(torch.float32).reshape(-1,3,3)
        positions, inds = merge_duplicate_verts(triPositions)
        print(f' - merged duplicate verts: {triPositions.shape[0] * triPositions.shape[1]} -> {positions.shape[0]}')
    else:
        positions = out2.to(torch.float32).reshape(-1,3)
        inds = torch.arange(positions.size(0)*positions.size(1)).to(torch.int32).view(-1,3)

    print('max ind', inds.max())
    print('len verts', len(positions))


    colors = np.ones((positions.shape[0],4), dtype=np.float32)
    colors[:,0] = positions[:,0]
    # normals = compute_normals(positions, inds)
    normals = positions / np.linalg.norm(positions,axis=1,keepdims=True)

    from render import GridRenderer, glDisable, GL_CULL_FACE, glEnable
    r = GridRenderer((1024,)*2)
    r.init(True)
    r.set_mesh(
            'hello',
            inds=inds,
            positions=positions,
            colors=colors,
            normals=normals,
            # verts=np.copy(out2.cpu().numpy(),'C'),
            # posRange=(0,3),
            # normalRange=(3,6),
            )
    glDisable(GL_CULL_FACE)
    # glEnable(GL_CULL_FACE)

    while True:
        r.startFrame()
        r.render()
        r.endFrame()

    return out2

def test_marching_tetra():
    SZ = 24
    coords = torch.cartesian_prod(*(torch.arange(SZ),)*3)
    # vals = coords[:, 0] / coords[:,0].max()
    # vals = coords[:, 0].float() / coords[:,0].max() #- .5 # Wall
    vals = -((coords - SZ/2).float().norm(dim=1) - 10) # Sphere
    # vals = (coords - (SZ+.5)/2).float().norm(dim=1) - .5

    st0 = torch.sparse_coo_tensor(coords.t(),vals, size=(SZ,)*3).coalesce()
    st = replicate_to_gridcells(st0)
    coords,vals = st.indices().t(), st.values()

    coords = coords.float() - SZ / 2
    gridScale = .06
    coords = coords * gridScale

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
    # marching_tetra(coords,vals, iso=.0)
    # vals = vals - vals.min()
    # vals = vals / vals.max()
    marching_tetra(coords,vals, iso=.4, gridScale=gridScale)
    sys.exit(0)


def merge_duplicate_verts(triVerts, powTwoResolution=20):
    assert triVerts.ndim == 3
    assert triVerts.size(1) == 3
    assert triVerts.size(2) == 3
    assert powTwoResolution <= 21
    NT = triVerts.size(0)
    N = NT*3

    vertsFlat = triVerts.view(N,3)
    SIZE = 1<<powTwoResolution
    offset = vertsFlat.min(0).values.unsqueeze(0)
    c = vertsFlat - offset
    scale = (SIZE-1) / c.max()
    c.mul_(scale)
    c = c.t().long()

    # NOTE: What I really need is thrust::sort_by_key() thrust::inclusive_scan_by_key() that simply keeps the first value.
    #       coalesce() sorts-by-key then unconditionally thrust::reduce_by_key()s with a sum operation.
    #       If I could even get it to do a min operation, I could get it to work. But any arithmetic trick seems to rely on
    #       an integer encoding that grows as n^n, making it unrealistic. I think you could do this in 3x3x3 subgrids, but
    #       that's pretty inelegant.

    '''
    b = (N+1)*(N+1)
    # v = 1+torch.arange(N,device=c.device,dtype=torch.int64) * b
    rng = torch.arange(N,device=c.device,dtype=torch.int64)
    v = (1+rng) + b ** (rng)

    t = torch.sparse_coo_tensor(c,v, size=(SIZE,)*3).coalesce()

    o_verts = t.indices().t().float()
    o_verts.div_(scale)
    o_verts.add_(offset)

    out = []
    left = t.values()
    ii = 0
    print('b',b)
    print('v',v)
    print('original',left)
    while len(left):
        # cur = left[(left>0) & (left < b)]
        cur = left
        print('ii',ii,'have',len(cur))
        if len(cur):
            cur = cur % b
            print('have ids', cur)
        left = left - b
        # left = left // b
        left = left[left>0]
        ii += 1
        if ii > 15: break
    '''

    # values grow as n^n, not possible.  :(
    '''
    # b = (N+1)*(N+1)
    N = 100
    b = (N+1)
    rng = torch.arange(N,device=c.device,dtype=torch.int64)
    # v = (1+rng) * (b ** (rng))
    v = (1+rng) * (b * (rng))
    # v = (1+rng) * ((rng) ** b)

    lv = v.log10() / np.log10(b)
    print(v)
    print(lv)
    '''


    # FIXME: cpu hash-map impl for now
    c = c.t().cpu().numpy().view(np.uint64) # [N,3]
    # cc = (c[:,2] << 60) | (c[:,1] << 40) | (c[:,0] & ((1<<20)-1))
    Z = 0b111111111111111111110000000000000000000000000000000000000000
    Y = 0b000000000000000000001111111111111111111100000000000000000000
    X = 0b000000000000000000000000000000000000000011111111111111111111
    cc = ((c[:,2] << 40)&Z) | ((c[:,1] << 20)&Y) | (c[:,0]&X)
    # cc = [tuple(a) for a in c]
    d = {}
    # print(N,cc.shape)
    for i,k in enumerate(cc):
        if k in d: d[k].append(i)
        else: d[k] = [i]

    o_verts = []
    o_tris = -np.ones((NT*3),dtype=np.int32)
    # print(d)
    for k,vertIdList in d.items():
        vi = len(o_verts)
        o_verts.append(vertsFlat[vertIdList[0]])
        for ii in vertIdList:
            o_tris[ii] = vi
    assert (o_tris >= 0).all()

    o_tris = torch.from_numpy(o_tris.reshape(-1,3).view(np.int32))
    o_verts = torch.stack(o_verts)

    # print(o_verts)
    # print(o_tris)

    return o_verts, o_tris

if 0:
    verts = torch.FloatTensor([
        0,0,0,
        .5,.5,0,
        .5,.0,0,

        .5,.0,0,
        .5,.5,0,
        1,1,0]).view(-1,3,3).cuda()
    merge_duplicate_verts(verts)
    exit()



test_marching_tetra()
