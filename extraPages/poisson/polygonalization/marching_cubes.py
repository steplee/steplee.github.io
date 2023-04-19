from .marching_common import *

# http://paulbourke.net/geometry/polygonise/
#
# This is a batched version of the above marching cubes algorithm.
#
# `positions` should be a floating point tensor [N,3] of 3d left-back-top points.
# `vals` should be a floating point tensor [N,8] of isovalues of each of the points and it's seven neighbours.
# The order should match that of `replicate_to_gridcells()`: a truth table in ZYX order.
#
def marching_cubes(positions, vals, iso=0, gridScale=1):
    N,D = positions.size(0), positions.device


    offset = positions.min(0).values
    positions = positions.float()
    # positions.sub_(offset)
    # positions.div_(scale)
    # vals.div_(scale)

    N = positions.size(0)
    assert positions.size(1) == 3
    assert vals.size() == (N,8)
    D = positions.device

    # Must be long -- cannot be byte. That would be interpreted as a mask later on.
    cubeIndex  = torch.zeros((N), dtype=torch.int64, device=D)
    for i in range(8):
        cubeIndex |= (1<<i) * (vals[:,i] < iso).long()

    # NOTE: My triTable differs a little bit from Bourke's
    # He uses a loop until any -1 is read.
    # Here I increment the ids of vertList and have the zero index be invalid.
    edgeTable = torch.from_numpy(marchingCubes_edgeTable).to(D)
    triTable  = torch.from_numpy(marchingCubes_triTable).to(D)[:,:15].view(256,5,3) + 1

    edges = edgeTable.gather(0, cubeIndex)
    tris  = triTable[cubeIndex]
    print(f' - tris  size {tris.shape}')

    p = torch.FloatTensor([
        0,0,0, 1,0,0, 1,1,0, 0,1,0,
        0,0,1, 1,0,1, 1,1,1, 0,1,1]).to(D).reshape(-1,3) * gridScale
    def blend(ai,bi):
        av = vals[:,ai]
        bv = vals[:,bi]
        ap = p[ai]
        bp = p[bi]

        mu = (iso - av) / (bv-av)
        mu.masked_fill_((iso-av).abs() < 1e-8, 0)
        mu.masked_fill_((iso-bv).abs() < 1e-8, 1)
        mu.masked_fill_((av-bv).abs() < 1e-8, 0)

        op = positions + ap + mu.view(-1,1)*(bp-ap)
        # op[~valid_mask] = -1
        return op

    vertList = torch.zeros((N,13,3),dtype=torch.float32,device=D)
    vertGroups = (
            (-1,-1),
            (0,1), (1,2), (2,3), (3,0), (4,5), (5,6),
            (6,7), (7,4), (0,4), (1,5), (2,6), (3,7) )
    for i,(ai,bi) in enumerate(vertGroups):
        # edge zero is made to be invalid (x=y=z=-1)
        if i == 0: vertList[:,i,:] = -1
        else:
            # ai = torch.full((N,),ai, dtype=torch.long, device=D)
            # bi = torch.full((N,),bi, dtype=torch.long, device=D)
            vertList[:,i,:] = blend(ai,bi)

    # I finally understand how gather works.
    out = torch.zeros((N,5,3,3),dtype=torch.float32)
    for triIdx in range(5):
        for j in range(3):
            out[:,triIdx,j] = vertList.gather(1,tris[:,triIdx,j].view(N,1,1).repeat(1,1,3))[:,0,:]

    # Array of triangle vertex components.
    out1 = out.view(-1,9)
    N0 = out1.size(0)
    # Remove invalid tris (should have 3x <-1,-,1-,1>)
    out1 = out1[(out1!=-1).all(1)]
    N1 = out1.size(0)
    out1 = out1.reshape(-1,3,3)
    print(f' - removed invalid tris: {N0} -> {N1}')

    out2 = out1
    # out2 = out2.mul_(scale)
    # out2 = out2.add_(offset.view(1,1,3))

    return out2
