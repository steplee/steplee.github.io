import torch, torch.nn, torch.nn.functional as F
import time

torch.manual_seed(0)



def printTimer(st, name):
    print(' - \'{:<12s}\' took {:.2f}ms'.format(name, 1000*(time.time()-st)))
def printElapsed(tt, name):
    print(' - \'{:<12s}\' took {:.2f}ms'.format(name, 1000*(tt)))

def q_to_matrix1(q):
    # r,i,j,k = q[0], q[1], q[2], q[3]
    # return torch.stack((
    r,i,j,k = q[0:1], q[1:2], q[2:3], q[3:4]
    return torch.cat((
        1-2*(j*j+k*k), 2*(i*j-k*r), 2*(i*k+j*r),
        2*(i*j+k*r), 1-2*(i*i+k*k), 2*(j*k-i*r),
        2*(i*k-j*r), 2*(j*k+i*r), 1-2*(i*i+j*j))).view(3,3)
def q_to_matrix(q):
    r,i,j,k = q[:,0], q[:,1], q[:,2], q[:,3]
    return torch.stack((
        1-2*(j*j+k*k),
        2*(i*j-k*r),
        2*(i*k+j*r),
        2*(i*j+k*r),
        1-2*(i*i+k*k),
        2*(j*k-i*r),
        2*(i*k-j*r),
        2*(j*k+i*r),
        1-2*(i*i+j*j)),1).view(-1,3,3)

@torch.jit.script # <- Improves speed of 'jacobian' by about 30%
def mapProject(poses, pts):
    q, t = poses[:,:4], poses[:,4:]
    pts = pts.view(pts.size(0), 1,3)

    tpts =  (q_to_matrix(q) @ (pts - t).permute(1,0,2).permute(0,2,1)).permute(0,2,1)
    # tpts =  (q_to_matrix(q).permute(0,2,1) @ (pts - t).permute(1,0,2).permute(0,2,1)).permute(0,2,1)
    return tpts[...,:2] / tpts[...,2:]

dev = torch.device('cpu')
# dev = torch.device('cuda')


def generate_data(nposes, npts):
    pts = torch.rand(npts,3) * 2 - 1
    '''
    truePoses = torch.tensor((
        1.,0,0,0, 0,0,-2,
        1.,0,0,0, 0,1,-2,
        1.,0,0,0, 1,0,-2)).view(-1,7)
    predPoses = torch.tensor((
        1.,0,0,0, 0,0,-2.1,
        1.,0,0,0, 0,1.1,-2,
        1.,0,0,0, 1.2,0,-2)).view(-1,7)
    '''

    truePoses = torch.rand(nposes, 7)
    truePoses[:,:4] = torch.tensor([1.,0,0,0])
    truePoses[:,4:] *= 2
    truePoses[:,4:] -= 1
    truePoses[:,-1] *= .1
    truePoses[:,-1] -= 2

    predPoses = truePoses.clone()
    predPoses[:, 4:] += torch.randn_like(predPoses[:,4:]) * torch.tensor((.5,.5,.1))

    pts = pts.to(dev)
    truePoses = truePoses.to(dev)
    predPoses = predPoses.to(dev)
    observations = mapProject(truePoses, pts)
    return pts, truePoses, predPoses, observations


def do_dense():

    ncams, npts = 3,5
    pts, true_poses, est_poses, observations = generate_data(nposes=ncams, npts=npts)

    def ba(*inputs):
        pts = torch.stack(inputs[:npts])
        poses = torch.stack(inputs[-ncams:])

        pred = mapProject(poses, pts)
        return pred - observations

    for i in range(4):
        t0 = time.time()

        # inputs = (pts, est_poses)
        inputs = (*pts, *est_poses)
        Js = torch.autograd.functional.jacobian(ba, inputs)
        res = ba(*inputs)
        print(' - Iteration',i,'mse',res.norm(dim=-1).mean())

        with torch.no_grad():
            numResidualBlocks = npts*ncams
            numStates = ncams*7 + npts*3
            # A will be sized 15 * 2 x numStates =
            #                 30 x 36
            # 3 5 2 3
            J = torch.cat([j.view(ncams*npts*2,-1) for j in Js], -1)
            H = J.T @ J + torch.eye(J.shape[1],device=dev) * .001 # NOTE: Prior
            g = J.T @ res.reshape(-1)

            d = torch.linalg.solve(H, g)
            d_pts = d[:npts*3].view(-1,3)
            d_poses = d[npts*3:].view(-1,7)
            # print(d_pts)
            # print(d_poses)

            for i,d_po in enumerate(d_poses):
                est_poses[i] -= d_po * 1
                est_poses[i,0:4] = F.normalize(est_poses[i,0:4], dim=0)
            for i,d_pt in enumerate(d_pts): pts[i] -= d_pt

        t1 = time.time()
        print(' - took {:.5f}ms'.format((t1-t0)*1000))

def jacobian(func, inputs, create_graph=False, strict=False):
    from torch.autograd.functional import _as_tuple, _check_requires_grad, _autograd_grad, _grad_postprocess, _tuple_postprocess, _grad_preprocess
    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jacobian")
    inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

    outputs = func(*inputs)
    is_outputs_tuple, outputs = _as_tuple(outputs,
                                          "outputs of the user-provided function",
                                          "jacobian")
    _check_requires_grad(outputs, "outputs", strict=strict)

    jacobian: Tuple[torch.Tensor, ...] = tuple()
    for i, out in enumerate(outputs):

        # mypy complains that expression and variable have different types due to the empty list
        jac_i: Tuple[List[torch.Tensor]] = tuple([] for _ in range(len(inputs)))  # type: ignore
        for j in range(out.nelement()):
            vj = _autograd_grad((out.reshape(-1)[j],), inputs,
                                retain_graph=True, create_graph=create_graph)

            for el_idx, (jac_i_el, vj_el, inp_el) in enumerate(zip(jac_i, vj, inputs)):
                if vj_el is not None:
                    if strict and create_graph and not vj_el.requires_grad:
                        msg = ("The jacobian of the user-provided function is "
                               "independent of input {}. This is not allowed in "
                               "strict mode when create_graph=True.".format(i))
                        raise RuntimeError(msg)
                    jac_i_el.append(vj_el)
                else:
                    if strict:
                        msg = ("Output {} of the user-provided function is "
                               "independent of input {}. This is not allowed in "
                               "strict mode.".format(i, el_idx))
                        raise RuntimeError(msg)
                    jac_i_el.append(torch.zeros_like(inp_el))

        jacobian += (tuple(torch.stack(jac_i_el, dim=0) for (el_idx, jac_i_el) in enumerate(jac_i)), )
        '''
        jacobian += (tuple(torch.stack(jac_i_el, dim=0).view(out.size()
                     + inputs[el_idx].size()) for (el_idx, jac_i_el) in enumerate(jac_i)), )
                     '''

    jacobian = _grad_postprocess(jacobian, create_graph)

    return _tuple_postprocess(jacobian, (is_outputs_tuple, is_inputs_tuple))

# Project one pt into one image
def mapProject1(pose, pt):
    q,t = pose[:4], pose[4:]

    tpt = q_to_matrix1(q).T @ (pt - t)
    # tpt = q_to_matrix(q.view(1,4))[0] @ (pt - t)
    return tpt[:2] / tpt[2]

# Project N pt into one image
def mapProject2(pose, pts):
    q,t = pose[...,:4], pose[...,4:]

    # print(q_to_matrix1(q).T.shape,pts.shape, (pts-t.view(1,3)).T.shape)
    # tpts = (q_to_matrix1(q).T @ (pts - t.view(1,3)).T).T
    tpts = (pts-t.view(-1,3)) @ q_to_matrix1(q)
    return tpts[...,:2] / tpts[...,2:]

def get_functorch_jac(test=False):
    from functorch import jacfwd, vmap
    from functorch.compile import aot_function, memory_efficient_fusion, ts_compile

    # vmapping over just points (one camera at a time)
    # m1 = vmap(mapProject1, (None,0))
    # d_m1 = vmap(jacfwd(mapProject1, (0,1)), (None,0))

    # pts, truePoses, predPoses, observations = generate_data(4,5)
    # a = m1(truePoses[0], pts)
    # d = d_m1(truePoses[0], pts)
    # print(a.shape)
    # print([d.shape for d in d])

    # vmapping over BOTH points and cameras
    #    (but requiring same #points in each camera)
    m2 = vmap(vmap(mapProject1, (None,0)), (0,None))
    # d_m2 = vmap(vmap(jacfwd(mapProject1, (0,1)), (None,0)), (0,None))
    d_m2 = vmap(vmap(jacfwd(mapProject1, (0,1)), (None,0)), (0,None))
    # d_m2 = memory_efficient_fusion(d_m2)
    # d_m2 = aot_function(d_m2, ts_compile) # Slower?

    # tt = time.time()
    # for i in range(100):
    # a = m2(truePoses, pts)
    # d = d_m2(truePoses, pts)
    # printTimer(tt, '100 runs')

    # print(a.shape)
    # print([d.shape for d in d])
    return m2, d_m2

#
# mapProject1 :: (H,7) x (H,3) -> (H,2)
#
#   mapProject2 :: (7)   x (H,3)   -> (H,2)
# v_mapProject2 :: (C,7) x (C,H,3) -> (C,H,2)
#
# (C,7) x (C,H,3) -> (C,H,2)
#
#
def get_functorch_jac_2():
    from functorch import jacfwd, vmap
    from functorch.compile import aot_function, memory_efficient_fusion, ts_compile

    # Each camera projects the SAME number of points

    # m2 = vmap(vmap(mapProject2, (None,0)), (0,None))
    m2 = vmap(mapProject2, (0,0))
    # d_m2 = vmap(vmap(jacfwd(mapProject1, (0,1)), (None,0)), (0,None))
    # d_m2 = vmap(jacfwd(mapProject2, (0,1)), (0,0), out_dims=(0,1))
    # d_m2 = vmap(jacfwd(mapProject2, (0,1)), (0,0)) # correct, but only for poses
    # d_m2 = vmap(jacfwd(mapProject2, (0,1)), (0,0))
    # d_m2 = jacfwd(vmap(mapProject2, (0,0)), (0,1))
    # d_m2 = vmap(vmap(jacfwd(mapProject2, (0,1)), (None,0)), (0,None))

    d_m2 = vmap(jacfwd(mapProject2, (0,1)), (0,0))

    return m2, d_m2



def do_sparse():

    # NOTE: Cool: pytorch can backprop (and even get jac) through a sparse MM. Not useful for this though
    '''
    ind = torch.tensor(ind).T
    val = torch.tensor(val).float()
    val = torch.autograd.Variable(val,True)
    t  = torch.sparse_coo_tensor(ind, val)

    b = torch.rand(t.size(0)).view(-1,1)
    b = torch.autograd.Variable(b,True)

    def do_mult(a,b):
        # return torch.sparse.mm(a,b)
        return torch.sparse.mm(a,b).mean()

    # torch.autograd.functional.jacobian(do_mult, (t,b))
    Js = jacobian(do_mult, (t,b))
    for J in Js:print('   J ',J.shape, type(J))

    J = Js[0].coalesce()[0]
    H = torch.sparse.mm(J.t(), J)
    print(H.shape,'->',H._nnz())
    print(H.to_dense())
    '''

    nposes, npts = 2,5
    nobs = 3
    # nposes, npts = 10,100

    print('')
    nstates = nposes*7 + npts*3
    pts, truePoses, predPoses, observations = generate_data(nposes,npts)
    # observations = observations[...,:nobs, :]

    idxPose = lambda i,j: 7*i + j
    idxPt   = lambda i,j: 7*nposes + 3*i + j

    ftMap, ftDMap = get_functorch_jac()
    ftMap2, ftDMap2 = get_functorch_jac_2()

    for iter in range(4):
        t0 = time.time()

        # Form graph
        ind,val = [],[]
        resCnt = 0
        res = []
        mse = 0

        # Say every camera sees every point
        tt = time.time()
        accTime = 0

        # Select either:
        #   first branch) Slow torch.autograd version
        #   faster      ) functorch vmap.vmap and jacfwd version
        if 0:
            for posei in range(nposes):
                # Linearize and insert data to graph
                for pti in range(npts):

                    t = time.time()
                    J_pose, J_pt = torch.autograd.functional.jacobian(mapProject, (predPoses[posei:posei+1], pts[pti:pti+1]))
                    accTime += time.time() - t

                    with torch.no_grad():
                        J_pose, J_pt = J_pose.squeeze(), J_pt.squeeze()
                        # if posei == 0 and pti == 0: print('here',J_pose,'\n',J_pt)

                        predPts = mapProject(predPoses[posei:posei+1], pts[pti:pti+1])[0,0]
                        res_ = predPts - observations[posei,pti]
                        res.append(res_)

                    for ri in range(2):
                        for k in range(7):
                            ind.append((resCnt+ri, idxPose(posei,k)))
                            val.append(J_pose[ri,k])
                        for k in range(3):
                            ind.append((resCnt+ri, idxPt(pti,k)))
                            val.append(J_pt[ri,k])
                    resCnt += 2
                    # print([J.squeeze().shape for J in Js])
            mse = torch.stack(res).norm(dim=-1).mean()
            printElapsed(accTime, 'JacobianCalc')
        else:

            print(' *******************************************')
            print(' This mode is not supported in this file, see ba2.py')
            print(' *******************************************')
            assert False

        printTimer(tt, 'GraphForm')


        # Add priors
        for i in range(nstates):
            if i < nposes*7:
                # Weak prior on pose translations, stronger on orientation
                val.append(.001 if i % 7 >= 4 else .01)
            else:
                # Strong prior on points
                val.append(.01)
            ind.append((resCnt,i))
            resCnt += 1
        res.append(torch.zeros(nstates))

        res = torch.cat(res).view(-1)
        ind = torch.tensor(ind).T
        val = torch.tensor(val)
        print(res.shape,ind.shape)


        tt = time.time()
        J = torch.sparse_coo_tensor(ind,val, size=(resCnt, nposes*7+npts*3))
        H = torch.sparse.mm(J.t(), J)
        g = J.t() @ res
        if iter == 0:
            print(' - Sparsity stats')
            print(' \t- J ::',J.shape,'(nnz {}, {:.2f} occ)'.format(J._nnz(), J._nnz() / J.numel()))
            print(' \t- H ::',H.shape,'(nnz {}, {:.2f} occ)'.format(H._nnz(), H._nnz() / H.numel()))
        printTimer(tt, 'SparseMult')

        tt = time.time()
        H = H.to_dense()
        printTimer(tt, 'Densify')
        tt = time.time()
        d = torch.linalg.solve(H, g)
        printTimer(tt, 'Solve')

        d_pts = d[nposes*7:].view(-1,3)
        d_poses = d[:nposes*7].view(-1,7)

        for i,d_po in enumerate(d_poses):
            predPoses[i] -= d_po * 1
            predPoses[i,0:4] = F.normalize(predPoses[i,0:4], dim=0)
        for i,d_pt in enumerate(d_pts): pts[i] -= d_pt

        # print(' - took {:.5f}ms'.format((time.time()-t0)*1000))
        printTimer(t0,'Iteration')
        print(' \t\t\t\t- Iteration',iter,'mse',mse)


def do_cg():
    pass




# do_dense()
do_sparse()
# get_functorch_jac(test=True)
# do_cg()
