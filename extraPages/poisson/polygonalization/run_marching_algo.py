from .marching_common import *
from .marching_cubes import marching_cubes
from .marching_tetra import marching_tetra

def test_marching(method='cubes'):
    SZ = 32
    coords = torch.cartesian_prod(*(torch.arange(SZ),)*3)
    # vals = coords[:, 0] / coords[:,0].max()
    # vals = coords[:, 0].float() / coords[:,0].max() #- .5 # Wall
    # vals = -(((coords - SZ/2).float()).norm(dim=1) - 10) # Sphere
    vals = -(((coords - SZ/2).float()/SZ + ((15*3.141*coords[:,1:2])**2/SZ).cos()/SZ + (15*3.141*coords[:,0:1]/SZ).cos()/SZ).norm(dim=1) - .3) # Sphere
    # vals = (coords - (SZ+.5)/2).float().norm(dim=1) - .5

    st0 = torch.sparse_coo_tensor(coords.t(),vals, size=(SZ,)*3).coalesce()
    st = replicate_to_gridcells(st0)
    coords,vals = st.indices().t(), st.values()

    gridScale = 1/SZ
    positions = (coords.float() - SZ / 2) * gridScale

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

    print(f' - input sizes: {positions.shape}, {vals.shape}')
    print('vals max',vals.max())
    print('vals min',vals.min())
    # vals = vals - vals.min()
    # vals = vals / vals.max()

    # -------------------------------------------------------------------------------------
    # Run polygonalization.
    # -------------------------------------------------------------------------------------
    # NOTE: A small non-zero isovalue gets rid of grid-borders apparenlty
    if method == 'cubes':
        out2 = marching_cubes(positions,vals, iso=1e-6, gridScale=gridScale)
    elif method == 'tetra':
        out2 = marching_tetra(positions,vals, iso=1e-6, gridScale=gridScale)
    else:
        assert False

    # -------------------------------------------------------------------------------------
    # Post Processing
    # -------------------------------------------------------------------------------------

    # NOTE: Control vertex reduction here.
    #       Also controls flat vs smooth normals!
    if 1:
        triPositions = out2.to(torch.float32).reshape(-1,3,3)
        positions, inds = merge_duplicate_verts(triPositions)
        print(f' - merged duplicate verts: {triPositions.shape[0] * triPositions.shape[1]} -> {positions.shape[0]}')
    else:
        positions = out2.to(torch.float32).reshape(-1,3)
        inds = torch.arange(positions.size(0)).to(torch.int32).view(-1,3)


    print('max ind', inds.max())
    print('len verts', len(positions))
    print(f'\n - Have {len(positions)} verts and {len(inds)} tris\n')

    normals = compute_normals(positions, inds)
    # normals = positions / np.linalg.norm(positions,axis=1,keepdims=True)

    colors = np.ones((positions.shape[0],4), dtype=np.float32)
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


test_marching('cubes')
test_marching('tetra')
