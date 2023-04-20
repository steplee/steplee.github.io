from .marching_common import *
from .marching_cubes import marching_cubes
from .marching_tetra import marching_tetra

def run_marching(isovalue_st0, isolevel=1e-6, method='cubes'):
    SZ = isovalue_st0.size()[0]
    assert isovalue_st0.size(0) == SZ
    assert isovalue_st0.size(1) == SZ
    assert isovalue_st0.size(2) == SZ

    coords1 = isovalue_st0.indices()
    vals1 = isovalue_st0.values()
    st1 = torch.sparse_coo_tensor(coords1,vals1, size=(SZ,)*3).coalesce()
    st = replicate_to_gridcells(st1)
    coords,vals = st.indices().t(), st.values()

    gridScale = 1./SZ
    gridOff = -SZ//2
    positions = (st.indices() + gridOff).float().t() * gridScale
    vals = st.values()

    # print(f' - input sizes: {positions.shape}, {vals.shape}')
    # print('vals max',vals.max())
    # print('vals min',vals.min())

    # -------------------------------------------------------------------------------------
    # Run polygonalization.
    # -------------------------------------------------------------------------------------
    # NOTE: A small non-zero isovalue gets rid of grid-borders apparenlty
    if method == 'cubes':
        marched = marching_cubes(positions,vals, iso=isolevel, gridScale=gridScale)
    elif method == 'tetra':
        marched = marching_tetra(positions,vals, iso=isolevel, gridScale=gridScale)
    elif method == 'surfaceNets':
        marched = marching_surface_net_multires(positions,vals, iso=isolevel, gridScale=gridScale)
    else:
        assert False

    # -------------------------------------------------------------------------------------
    # Post Processing
    # -------------------------------------------------------------------------------------

    # NOTE: Control vertex reduction here.
    #       Also controls flat vs smooth normals!
    if 1:
        triPositions = marched.to(torch.float32).reshape(-1,3,3)
        positions, inds = merge_duplicate_verts(triPositions)
        print(f' - merged duplicate verts: {triPositions.shape[0] * triPositions.shape[1]} -> {positions.shape[0]}')
    else:
        positions = marched.to(torch.float32).reshape(-1,3)
        inds = torch.arange(positions.size(0)).to(torch.int32).view(-1,3)

    return positions, inds, (gridOff,gridScale)

# Run a viz showing the mesh, after computing normals and using them as colors.
def show_marching_viz(positions, triInds, pts=None, nrls=None):
    inds = triInds

    normals = compute_normals(positions, inds)
    # normals = positions / np.linalg.norm(positions,axis=1,keepdims=True)

    colors = torch.ones((positions.shape[0],4), dtype=torch.float32, device=positions.device)
    # colors[:,0:3] = positions[:,0:3].abs() * 4
    colors[:,0:3] = normals[:,0:3].abs() * 1

    # -------------------------------------------------------------------------------------
    # Viz
    # -------------------------------------------------------------------------------------

    from render import GridRenderer, glDisable, GL_CULL_FACE, glEnable
    r = GridRenderer((1024,)*2)
    r.init(True)
    r.set_mesh(
            'hello',
            inds=inds,
            positions=positions,
            colors=colors,
            normals=normals,
            wireframe=True)
    if pts is not None:
        r.set_points_and_normals('ptsNrls',pts=pts.cpu().numpy(),nrls=nrls.cpu().numpy())
    # glDisable(GL_CULL_FACE)
    glEnable(GL_CULL_FACE)

    while True:
        r.startFrame()
        r.render()
        r.endFrame()
        if r.q_pressed: break

def test_marching(method='cubes'):
    SZ = 32
    coords = torch.cartesian_prod(*(torch.arange(SZ),)*3).cuda()
    # vals = coords[:, 0] / coords[:,0].max()
    # vals = coords[:, 0].float() / coords[:,0].max() #- .5 # Wall
    # vals = -(((coords - SZ/2).float()).norm(dim=1) - 10) # Sphere
    vals = -(((coords - SZ/2).float()/SZ + ((15*3.141*coords[:,1:2])**2/SZ).cos()/SZ + (15*3.141*coords[:,0:1]/SZ).cos()/SZ).norm(dim=1) - .3) # Sphere
    # vals = (coords - (SZ+.5)/2).float().norm(dim=1) - .5

    st0 = torch.sparse_coo_tensor(coords.t(),vals, size=(SZ,)*3).coalesce()
    positions, inds, gridXform = run_marching(st0, method=method)


    print('max ind', inds.max())
    print('len verts', len(positions))
    print(f'\n - Have {len(positions)} verts and {len(inds)} tris\n')

    normals = compute_normals(positions, inds)
    # normals = positions / np.linalg.norm(positions,axis=1,keepdims=True)

    colors = torch.ones((positions.shape[0],4), dtype=torch.float32, device=positions.device)
    # colors[:,0:3] = positions[:,0:3].abs() * 4
    colors[:,0:3] = normals[:,0:3].abs() * 1

    # -------------------------------------------------------------------------------------
    # Viz
    # -------------------------------------------------------------------------------------

    from render import GridRenderer, glDisable, GL_CULL_FACE, glEnable
    r = GridRenderer((1024,)*2)
    r.init(True)
    r.set_mesh(
            'hello',
            inds=inds,
            positions=positions,
            colors=colors,
            normals=normals,
            wireframe=True
            )
    # glDisable(GL_CULL_FACE)
    glEnable(GL_CULL_FACE)

    while True:
        r.startFrame()
        r.render()
        r.endFrame()
        if r.q_pressed: break


if __name__ == '__main__':
    test_marching('cubes')
    test_marching('tetra')
    test_marching('surfaceNets')
