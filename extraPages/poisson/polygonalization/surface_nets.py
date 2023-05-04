import torch
from .marching_common import replicate_to_gridcells

# I want to implement a surface-nets like approach because it's pretty hard
# to do marching cubes in a multi-resolution domain.
# Surface nets (or any dual approach like it) should be easier, I think.

# FIXME: I think you need to base this on quads, not tris.
# https://0fps.net/2012/07/12/smooth-voxel-terrain-part-2/
# https://www.boristhebrave.com/2018/04/15/dual-contouring-tutorial/

# To find output triangles, we need to examine triplets of created vertices.
# The candidate triplets are those that are within a distance of one from the center vertex.
# This can be done, again, as a convolution. It has the following constraints:
#      1) The other two verts must be within distance one from the first.
#      2) The other two verts must be within distance one from each other.
#      3) The other two verts must not be the same (would be degenerate triangle)
#      4) Since we're running at all points, we can say that this pos must be the lowest one.
#
# Since the conv is done at every vertex as the center, only two loops are needed.
#

# This is close, but over-connected.
# New approach below ... don't use this.
def find_tris(sparseTensor):
    c = sparseTensor.indices()
    # v = sparseTensor.values()
    N,D = c.size(1), c.device
    S = sparseTensor.size(0)
    assert sparseTensor.size(1) == S and sparseTensor.size(2) == S

    # Dense-dim + sparse-dim
    # The values will be integers telling which idx vertices a triangle consists of.
    # The idxs are offset by one. This way any being 0 means invalid triangle.
    size = (*sparseTensor.size(), 3)
    # out = torch.sparse_coo_tensor(c,ov,size=size).coalesce()


    origIds = torch.arange(N, device=D).view(-1,1).repeat(1,3) + 1 # offset by one
    select0 = torch.FloatTensor((1,0,0)).to(D).view(1,3).long()
    select1 = torch.FloatTensor((0,1,0)).to(D).view(1,3).long()
    select2 = torch.FloatTensor((0,0,1)).to(D).view(1,3).long()

    print(c.shape)
    origSt = torch.sparse_coo_tensor(c, torch.ones((N,3),device=D,dtype=torch.long), size=size).coalesce()
    print(origSt.shape, origSt.dtype)

    print(c.size(), origIds.size(), select0.size())
    st0 = torch.sparse_coo_tensor(c, origIds*select0, size=size).coalesce()

    # grid = torch.cartesian_prod(*(torch.arange(-1,2),)*3)
    grid = torch.cartesian_prod(*(torch.arange(0,2),)*3)

    # Output indices
    triInds = torch.empty((0,3), dtype=torch.long, device=D)

    for idx1, (ax,ay,az) in enumerate(grid):
        for idx2, (bx,by,bz) in enumerate(grid):
            if (ax!=0 or ay!=0 or az!=0) and \
               (bx!=0 or by!=0 or bz!=0) and \
               abs(bx-ax)+abs(by-ay)+abs(bz-az) <= 2 and \
               (idx1!=idx2):
                off1 = torch.LongTensor([ax,ay,az]).to(c.device).view(3,1)
                off2 = torch.LongTensor([bx,by,bz]).to(c.device).view(3,1)

                coff1 = c - off1
                coff2 = c - off2
                mask1 = (coff1>=0).all(0) & (coff1<S).all(0)
                mask2 = (coff2>=0).all(0) & (coff2<S).all(0)
                coff1 = coff1[:,mask1]
                coff2 = coff2[:,mask2]

                vals1 = origIds[mask1] * select1
                vals2 = origIds[mask2] * select2

                st1 = (origSt * torch.sparse_coo_tensor(coff1, vals1, size=size).coalesce())
                st2 = (origSt * torch.sparse_coo_tensor(coff2, vals2, size=size).coalesce())

                news = (st0+st1+st2).coalesce()
                news_c = news.indices()
                news_v = news.values()

                news_v = news_v[(news_v!=0).all(1)] - 1

                # triInds = torch.cat((triInds, (news_v!=0).all(1)))
                # print(triInds.shape, news.indices().shape)
                triInds = torch.cat((triInds, news_v), 0)

    print(f' - out triInds shape {triInds.shape} {triInds.dtype}')

    assert (triInds < N).all()

    print(triInds.shape, triInds.numel(), N)
    tmp = torch.sparse_coo_tensor(triInds.view(1,-1), torch.ones(triInds.numel(),device=D), size=(N,)).coalesce()
    kept = tmp.indices()[0,tmp.values() > 0]
    vertexKeepMask = torch.zeros(N, dtype=torch.bool, device=D)
    vertexKeepMask[kept] = 1
    print(f' - keeping {vertexKeepMask.sum().item()} / {vertexKeepMask.size()}')

    return vertexKeepMask, triInds

#
# This is close, but no cigar!
# You end up with a honeycomb like pattern.
# Could perhaps fix by looking at normal of isofunction, or some other way?
# Also, all tris are replicated at least twice.
#
def surface_nets_one_res_0(indices, vals, iso=0, gridScale=1):
    N,D = indices.size(1), indices.device

    assert indices.size(0) == 3
    assert vals.size() == (N,8)
    D = indices.device

    cmp = (vals < iso)
    masks1 = (cmp[:,1:] != cmp[:,0:1]).any(1)

    vertIndices = indices[:,masks1]
    print(f'vertIndices shape {vertIndices.shape} from indices {indices.shape} vals {vals.shape}')
    # print(vertIndices)
    vertSt = torch.sparse_coo_tensor(vertIndices, torch.ones(vertIndices.size(1), device=D)).coalesce()
    print(vertSt.shape)
    # vertNeighbors = replicate_to_gridcells(vertSt)

    # Create triangles for each group-of-three
    vertexKeepMask, triInds = find_tris(vertSt)

    if 1:
        NV0 = vertSt.indices().size(1)
        # remap = torch.arange(NV0, device=D)[vertexKeepMask]
        # remap = (vertexKeepMask.long().cumsum(0))
        # Actually, we want a *postfix* scan, not a prefix scan. So subtract one.
        remap = (vertexKeepMask.long().cumsum(0) - 1).clamp(0,9999999999)

        vertIndices = vertIndices[:, vertexKeepMask]
        print(f'remapping num verts {NV0} -> {vertexKeepMask.sum().item()}')
        print(remap)
        # print(remap[triInds])
        # return None
        triInds = remap[triInds]
        print(f' - final verts & inds {vertIndices.shape} {triInds.shape}')

    NV1 = vertIndices.size(1)
    assert (triInds < NV1).all()


    return vertIndices, triInds












# FIXME: The mikola lysenko post recommends taking mean of marching-cubes based cell vertices,
#        while using the connectivity of the voxelized approach.
#        Doing that requires the original sparse tensor, so I'd have to change a bit here.


def surface_nets_one_res_1(positions, values, iso=0, gridScale=1):
    N,D = positions.size(0), positions.device
    # size = positions.size()[:3]
    # assert positions.values().size() == (N,4)

    # For all 6 face-neighbors, if the isovalue cross create a QUAD spanning those two.
    # So yea, not too bad.
    # Afterward, sweep up the duplicate verts like before

    # Only make quad if I have lower coordinate (in each axis) then the neighbor.
    # That'll prevent duplicates. FIXME: But it will break the lower-left-bottom edge.

    # inds = faceneighbors.indices()
    # inside = faceneighbors.values() < iso
    inside = values < iso
    # inside &= values[:,0:1] != 0 # make exactly '0.0' not create faces.

    d_ = torch.FloatTensor((
        -.5,-.5,0,
        .5,-.5,0,
        .5,.5,0,
        -.5,.5,0)).view(4,3).to(D)
    # d_ += .5
    d_ *= gridScale

    vs = torch.empty((0,3),dtype=torch.float32,device=D)

    # for i in range(1):
    # for i in (0,3):
    # for i in (2,5):
    for i in range(6):
        if i <  3: dx,dy,dz = ( int(i==j) for j in range(3))
        if i >= 3: dx,dy,dz = ( int(i-3==j) for j in range(3))
        # if i <  3: dz,dy,dx = ( int(i==j) for j in range(3))
        # if i >= 3: dz,dy,dx = ( int(i-3==j) for j in range(3))
        dd = -torch.FloatTensor((dx,dy,dz)).to(D).view(1,3)
        if i>=3: dd *= -1
        print('dd',dd, dx,dy,dz)

        if dx !=0: order = (2,0,1)
        if dy !=0: order = (1,2,0)
        if dz !=0: order = (0,1,2)
        # if i >= 3: order = (order[0], order[2], order[1])

        # mask = (inside[:,0] != inside[:,1+i])
        mask = (inside[:,0] == 1) & (inside[:,1+i] == 0)
        print(f' - mask on {mask.sum().item()} / {mask.size(0)}')

        # p0 = inds[:,mask].t().float() + dd * gridScale * .5
        # p0 = positions[mask] + dd * gridScale * .5
        p0 = positions[mask] + dd * gridScale * .5
        print(p0.shape, d_[0:1,order].shape)
        p1 = p0 + d_[0:1,order]
        p2 = p0 + d_[1:2,order]
        p3 = p0 + d_[2:3,order]
        p4 = p0 + d_[3:4,order]
        if i < 3:
            ps = torch.cat((p1,p2,p3, p3,p4,p1), 1).view(-1,3)
        else:
            ps = torch.cat((p2,p1,p3, p4,p3,p1), 1).view(-1,3)
        vs = torch.cat((vs, ps), 0)

    return vs


