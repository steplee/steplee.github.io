from .marching_common import *

# Returns up to two tris per input.
# Should be called 6 times per grid cell
def marching_tetra_tri(positions, vals, scale,  iso, v0,v1,v2,v3, swap):
    N = positions.size(0)
    assert positions.size(1) == 3
    assert vals.size() == (N,8)
    D = positions.device

    # Must be long -- cannot be byte. That would be interpreted as a mask later on.
    triIndex  = torch.zeros((N), dtype=torch.int64, device=D)
    triIndex |= (1) * (vals[:,v0] < iso).long()
    triIndex |= (2) * (vals[:,v1] < iso).long()
    triIndex |= (4) * (vals[:,v2] < iso).long()
    triIndex |= (8) * (vals[:,v3] < iso).long()
    # print(' - triIndex shape',triIndex.shape)

    # This replaces the switch statement in `PolygoniseTri`
    # This is more like Bourke's marching cubes impl.
    # It will allow us to implement the procedure in batch fashion with pytorch.
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

    #
    # NOTE: I had to fix winding order of triangles. It was tedious.
    # It was necessary to selectively swap some groups-of-4-corners, hence the `swap` argument.
    # I had to go through each 16 and each of the 6 groups-of-4. Luckily it factored out given the swap argument.
    # NOTE: This could be avoided by re-arranging the table and groups-of-4.
    #
    if 0:
        for k,v in table1.items(): table[k, 0, :, :] = torch.LongTensor(v).to(D).view(3, 2)
        for k,v in table2.items(): table[k, 1, :, :] = torch.LongTensor(v).to(D).view(3, 2)
    else:
        for k,v in table1.items():
            if swap ^ (k in (0x3,0x7,0x9,0xA,0xB,0xD,0xE)): table[k, 0, :, :] = torch.LongTensor(v).to(D).view(3, 2)
            else:        table[k, 0, :, :] = torch.LongTensor(v).to(D).view(3, 2)[[1,0,2]]
        for k,v in table2.items():
            if swap ^ (k in (0x3,0x5,0x9)): table[k, 1, :, :] = torch.LongTensor(v).to(D).view(3, 2)[[1,0,2]]
            else:        table[k, 1, :, :] = torch.LongTensor(v).to(D).view(3, 2)
    # print('table\n',table)
    # for k in range(16): # if k != 0xC: table[k] = -1 table[k,0]=-1

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
        mu.masked_fill_((iso-av).abs() < 1e-8, 0)
        mu.masked_fill_((iso-bv).abs() < 1e-8, 1)
        mu.masked_fill_((av-bv).abs() < 1e-8, 0)

        op = positions + ap + mu.view(-1,1)*(bp-ap)
        op[~valid_mask] = -1
        return op

    for triId in range(2):
        for vertexId in range(3):
            out[:,triId,vertexId,:] = blend(table[triIndex, triId, vertexId, 0], table[triIndex, triId, vertexId, 1])

    # NOTE: The output consists of plenty of <-1,-1,-1> points which should be removed.
    return out

# http://paulbourke.net/geometry/polygonise/source1.c
#
# This is a batched version of the above marching tetrahedra algorithm.
#
# `positions` should be a floating point tensor [N,3] of 3d left-back-top points.
# `vals` should be a floating point tensor [N,8] of isovalues of each of the points and it's seven neighbours.
# The order should match that of `replicate_to_gridcells()`: a truth table in ZYX order.
#
def marching_tetra(positions, vals, iso=0, gridScale=1):
    N,D = positions.size(0), positions.device


    offset = positions.min(0).values
    positions = positions.float()
    # positions.sub_(offset)
    # positions.div_(scale)
    # vals.div_(scale)

    groups = (
        (False, (0,2,3,7)),
        (True , (0,2,6,7)),
        (False, (0,4,6,7)),
        (False, (0,6,1,2)),
        (True , (0,6,1,4)),
        (False, (5,6,1,4)),
    )

    # o = marching_tetra_tri(positions, vals, iso, 0,2,3,7)
    # print(f' - positions {positions.min(0).values} -> {positions.max(0).values}')
    # print(' - using grid scale', 1/gridScale)
    out = torch.zeros((N,len(groups),2,3,3), device=D)
    for i,(swap,grp) in enumerate(groups):
        out[:, i] = marching_tetra_tri(positions, vals, gridScale, iso, *grp, swap=swap)

    # Array of triangle vertex components.
    out1 = out.view(-1,9)
    N0 = out1.size(0)
    # Remove invalid tris (should have 3x <-1,-,1-,1>)
    out1 = out1[(out1!=-1).all(1)]
    N1 = out1.size(0)
    out1 = out1.reshape(-1,3,3)
    # print(out1)
    print(f' - removed invalid tris: {N0} -> {N1}')

    out2 = out1
    # out2 = out2.mul_(scale)
    # out2 = out2.add_(offset.view(1,1,3))

    return out2
